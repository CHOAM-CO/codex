//! Complete a requested history range from external tool records already in its rollout.
//!
//! Older writers stored AGY calls/results without materialized lifecycle items. Read only the
//! requested turns, and add their existing call identities to the existing history projection.

use std::collections::HashMap;
use std::collections::HashSet;

use codex_app_server_protocol::ThreadItem;
use codex_app_server_protocol::project_external_tool_call;
use codex_protocol::models::ResponseItem;
use codex_rollout::ResponseItemEnvelope;
use codex_rollout::ReverseJsonlScanner;
use codex_rollout::RolloutItem;
use codex_rollout::ScanOutcome;

use super::super::LocalThreadStore;
use super::super::rollout_lineage::RolloutLineage;
use super::sqlite_integer;
use super::thread_history_error;
use crate::ThreadStoreResult;

struct ExternalCall {
    call: ResponseItemEnvelope,
    result: Option<ResponseItemEnvelope>,
    ordinal: u64,
    updated_ordinal: u64,
    created_at_ms: i64,
    materialized: bool,
}

/// `range` bounds the original timeline ordinals; `turn_id` restricts an ordinary item query.
pub(super) async fn backfill(
    store: &LocalThreadStore,
    lineage: &RolloutLineage,
    turn_id: Option<&str>,
    range: Option<(u64, u64)>,
) -> ThreadStoreResult<()> {
    let pool = store.thread_history_db().await?;
    for segment in lineage.segments() {
        let projection_end = super::projection_state(store, segment.rollout_id())
            .await?
            .map(|state| state.next_byte_offset);
        let segment_start = sqlite_integer(segment.start_ordinal(), "rollout ordinal")?;
        let segment_end = segment
            .end_ordinal()
            .map(|ordinal| sqlite_integer(ordinal, "rollout ordinal"))
            .transpose()?;
        let rows = sqlx::query_as::<_, (String, i64, Option<i64>)>(
            r#"
SELECT turn_id, rollout_ordinal, rollout_end_byte_offset
FROM thread_turns
WHERE thread_id = ?
  AND rollout_ordinal >= ?
  AND (? IS NULL OR rollout_ordinal < ?)
  AND (? IS NULL OR turn_id = ?)
  AND (? IS NULL OR rollout_ordinal <= ?)
  AND (? IS NULL OR rollout_end_ordinal IS NULL OR rollout_end_ordinal >= ?)
            "#,
        )
        .bind(segment.rollout_id().to_string())
        .bind(segment_start)
        .bind(segment_end)
        .bind(segment_end)
        .bind(turn_id)
        .bind(turn_id)
        .bind(
            range
                .map(|(_, end)| sqlite_integer(end, "rollout ordinal"))
                .transpose()?,
        )
        .bind(
            range
                .map(|(_, end)| sqlite_integer(end, "rollout ordinal"))
                .transpose()?,
        )
        .bind(
            range
                .map(|(start, _)| sqlite_integer(start, "rollout ordinal"))
                .transpose()?,
        )
        .bind(
            range
                .map(|(start, _)| sqlite_integer(start, "rollout ordinal"))
                .transpose()?,
        )
        .fetch_all(pool)
        .await
        .map_err(thread_history_error)?;
        for (turn_id, start_ordinal, end_offset) in rows {
            let path = segment.rollout_path.clone();
            let segment_end = segment.end;
            let start_ordinal = u64::try_from(start_ordinal).map_err(thread_history_error)?;
            let end_offset = end_offset
                .map(u64::try_from)
                .transpose()
                .map_err(thread_history_error)?
                .map(|offset| segment_end.map_or(offset, |end| offset.min(end.end_byte_offset)))
                .or(segment_end.map(|end| end.end_byte_offset))
                .or(projection_end);
            let calls =
                tokio::task::spawn_blocking(move || -> ThreadStoreResult<Vec<ExternalCall>> {
                    let file = codex_rollout::open_rollout_seekable_reader(&path)
                        .map_err(thread_history_error)?;
                    let mut scanner = match end_offset {
                        Some(end) => ReverseJsonlScanner::new_at(file, end),
                        None => ReverseJsonlScanner::new(file),
                    }
                    .map_err(thread_history_error)?;
                    let mut records = Vec::new();
                    let mut materialized_ids = HashSet::new();
                    while let Some(record) = scanner
                        .scan_next_rollout_line()
                        .map_err(thread_history_error)?
                    {
                        let line = match record {
                            ScanOutcome::Parsed(line) => line,
                            ScanOutcome::Rejected(error) => {
                                return Err(thread_history_error(format!(
                                    "invalid rollout record in external tool history range: {error}"
                                )));
                            }
                        };
                        let ordinal = line.ordinal.ok_or_else(|| {
                            thread_history_error(
                                "external tool history range has no rollout ordinal",
                            )
                        })?;
                        if ordinal < start_ordinal {
                            break;
                        }
                        match line.item {
                            RolloutItem::ResponseItem(envelope)
                                if matches!(&envelope.item,
                            ResponseItem::FunctionCall { namespace, .. }
                            | ResponseItem::FunctionCallOutput { namespace, .. }
                            if namespace.as_deref() == Some("antigravity")) =>
                            {
                                records.push((ordinal, line.timestamp, envelope));
                            }
                            RolloutItem::EventMsg(
                                codex_protocol::protocol::EventMsg::ItemCompleted(event),
                            ) => {
                                materialized_ids
                                    .insert(ThreadItem::from(event.item).id().to_owned());
                            }
                            _ => {}
                        }
                    }
                    records.reverse();
                    let mut calls: HashMap<String, ExternalCall> = HashMap::new();
                    for (ordinal, timestamp, envelope) in records {
                        match &envelope.item {
                            ResponseItem::FunctionCall { call_id, .. } => {
                                if materialized_ids.contains(call_id)
                                    && !envelope.metadata.as_ref().is_some_and(|metadata| {
                                        metadata.external_executor_tool.is_some()
                                    })
                                {
                                    continue;
                                }
                                let created_at_ms =
                                    chrono::DateTime::parse_from_rfc3339(&timestamp)
                                        .map_err(thread_history_error)?
                                        .timestamp_millis();
                                let existing =
                                    calls
                                        .entry(call_id.clone())
                                        .or_insert_with(|| ExternalCall {
                                            call: envelope.clone(),
                                            result: None,
                                            ordinal,
                                            updated_ordinal: ordinal,
                                            created_at_ms,
                                            materialized: materialized_ids.contains(call_id),
                                        });
                                existing.call = envelope;
                                existing.updated_ordinal = ordinal;
                            }
                            ResponseItem::FunctionCallOutput {
                                call_id: Some(call_id),
                                ..
                            } => {
                                if let Some(call) = calls.get_mut(call_id) {
                                    call.result = Some(envelope);
                                    call.updated_ordinal = ordinal;
                                }
                            }
                            _ => {}
                        }
                    }
                    Ok(calls.into_values().collect())
                })
                .await
                .map_err(thread_history_error)??;
            if calls.is_empty() {
                continue;
            }
            let mut transaction = pool.begin().await.map_err(thread_history_error)?;
            for call in calls {
                let Some(item) = project_external_tool_call(&call.call, call.result.as_ref())
                else {
                    continue;
                };
                let item_json = serde_json::to_string(&item).map_err(thread_history_error)?;
                sqlx::query(
                    r#"
INSERT INTO thread_items (
    thread_id, turn_id, item_id, rollout_ordinal, updated_at_ordinal,
    created_at_ms, started_at_ms, completed_at_ms, item_type, item_json
) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, 'mcpToolCall', ?)
ON CONFLICT(thread_id, turn_id, item_id) DO UPDATE SET
    updated_at_ordinal = MAX(thread_items.updated_at_ordinal, excluded.updated_at_ordinal),
    item_json = CASE WHEN ? THEN
        json_set(thread_items.item_json, '$.externalExecutor', json_extract(excluded.item_json, '$.externalExecutor'))
        ELSE excluded.item_json END
WHERE json_extract(thread_items.item_json, '$.server') = 'antigravity'
                    "#,
                )
                .bind(segment.rollout_id().to_string())
                .bind(&turn_id)
                .bind(item.id())
                .bind(sqlite_integer(call.ordinal, "rollout ordinal")?)
                .bind(sqlite_integer(call.updated_ordinal, "rollout ordinal")?)
                .bind(call.created_at_ms)
                .bind(item_json)
                .bind(call.materialized)
                .execute(&mut *transaction)
                .await
                .map_err(thread_history_error)?;
            }
            transaction.commit().await.map_err(thread_history_error)?;
        }
    }
    Ok(())
}
