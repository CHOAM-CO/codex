//! Saves terminal speech in model and client history before realtime closure, without inference.

use crate::session::new_submission_id;
use crate::session::session::Session;
use codex_history::{CodexHarnessMetadata, RealtimeTranscriptTailSource, ResponseItemEnvelope};
use codex_protocol::items::TurnItem;
use codex_protocol::items::UserMessageItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::HasLegacyEvent;
use codex_protocol::protocol::ItemCompletedEvent;
use codex_protocol::protocol::TurnCompleteEvent;
use codex_protocol::protocol::TurnStartedEvent;
use codex_protocol::user_input::UserInput;
use codex_rollout::RolloutItem;
use codex_thread_store::PersistContext;
use std::sync::Arc;

#[expect(
    clippy::await_holding_invalid_type,
    reason = "record history before the owning task can finish or a new task can start"
)]
pub(super) async fn record(
    session: &Session,
    realtime_submission_id: String,
    text: String,
) -> std::io::Result<()> {
    let mut recording_context = session.new_inject_items_context().await;
    Arc::get_mut(&mut recording_context)
        .ok_or_else(|| std::io::Error::other("realtime transcript context is shared"))?
        .sub_id = new_submission_id();
    let active = session.active_turn.lock().await;
    let running_turn = active
        .as_ref()
        .and_then(|turn| turn.task.as_ref())
        .map(|task| Arc::clone(&task.turn_context));
    let recording_turn = running_turn.is_none();
    let turn = running_turn.unwrap_or(recording_context);
    let content = vec![UserInput::Text {
        text,
        text_elements: Vec::new(),
    }];
    let now = chrono::Utc::now();
    if recording_turn {
        let started = EventMsg::TurnStarted(TurnStartedEvent {
            turn_attribution: None,
            turn_id: turn.sub_id.clone(),
            root_turn_id: Some(turn.sub_id.clone()),
            trace_id: turn.trace_id.clone(),
            started_at: Some(now.timestamp()),
            model_context_window: turn.model_context_window(),
            collaboration_mode_kind: turn.mode(),
        });
        session
            .try_persist_rollout_items(&[RolloutItem::EventMsg(started)])
            .await?;
        session
            .record_started_turn(&turn.sub_id, /*attribution*/ None)
            .await;
    }
    let mut item = session.response_item_from_user_input(content.clone());
    Session::assign_missing_response_item_id(&mut item);
    let item_id = item
        .id()
        .expect("transcript response item has an ID")
        .to_string();
    let order = session.reserve_user_input_order().await;
    session
        .record_realtime_transcript_item(
            &turn,
            ResponseItemEnvelope {
                item,
                metadata: Some(CodexHarnessMetadata {
                    user_input_order: Some(order),
                    realtime_transcript_tail: Some(RealtimeTranscriptTailSource {
                        thread_id: session.thread_id,
                        turn_id: turn.sub_id.clone(),
                        realtime_submission_id,
                    }),
                    ..Default::default()
                }),
            },
        )
        .await?;

    // ItemCompleted is the existing source for thread/items/list. Keep the legacy
    // equivalent too; the store selects the appropriate format for this thread.
    let mut user_item = UserMessageItem::new(&content);
    user_item.id = item_id;
    let completed = EventMsg::ItemCompleted(ItemCompletedEvent {
        thread_id: session.thread_id,
        turn_id: turn.sub_id.clone(),
        item: TurnItem::UserMessage(user_item),
        started_at_ms: Some(now.timestamp_millis()),
        completed_at_ms: now.timestamp_millis(),
    });
    let mut events = Vec::new();
    let legacy = completed.as_legacy_events(/*show_raw_agent_reasoning*/ false);
    events.push(completed);
    events.extend(legacy);
    if recording_turn {
        events.push(EventMsg::TurnComplete(TurnCompleteEvent {
            turn_id: turn.sub_id.clone(),
            root_turn_id: Some(turn.sub_id.clone()),
            last_agent_message: None,
            error: None,
            started_at: Some(now.timestamp()),
            completed_at: Some(now.timestamp()),
            duration_ms: Some(0),
            time_to_first_token_ms: None,
        }));
    }
    // Persist the complete history record without advertising a running UI turn
    // or completing the model task that may still be streaming.
    session
        .try_persist_rollout_items(
            &events
                .into_iter()
                .map(RolloutItem::EventMsg)
                .collect::<Vec<_>>(),
        )
        .await?;
    drop(active);
    session
        .try_ensure_rollout_materialized(PersistContext::Standard)
        .await?;
    session.flush_rollout().await
}
