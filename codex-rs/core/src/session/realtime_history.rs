use super::session::Session;
use crate::realtime_history::RealtimeEventEffects;
use codex_history::RolloutItem;
use codex_protocol::protocol::Event;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::RealtimeConversationRealtimeEvent;
use codex_protocol::protocol::RealtimeEvent;
use codex_protocol::realtime::RealtimeItemContent;

impl Session {
    pub(super) async fn report_realtime_history_failure(
        &self,
        submission_id: &str,
        error: &anyhow::Error,
    ) {
        // The reducer lock is held by the caller; deliver this warning without re-entering it.
        let message = format!("failed to persist realtime history: {error:#}");
        let warning = EventMsg::Warning(codex_protocol::protocol::WarningEvent { message });
        self.persist_rollout_items(&[RolloutItem::EventMsg(warning.clone())])
            .await;
        self.deliver_event_raw(Event {
            id: submission_id.to_owned(),
            msg: warning,
        })
        .await;
    }

    pub(super) async fn send_realtime_history_effects(
        &self,
        submission_id: &str,
        effects: RealtimeEventEffects,
    ) -> anyhow::Result<()> {
        let event = |payload| Event {
            id: submission_id.to_string(),
            msg: EventMsg::RealtimeConversationRealtime(RealtimeConversationRealtimeEvent {
                payload,
            }),
        };
        if let Some(stream) = effects.transcript_stream {
            if let Some(item) = stream.started_item {
                self.deliver_event_raw(event(RealtimeEvent::HistoryItemStarted(item)))
                    .await;
            }
            self.deliver_event_raw(event(RealtimeEvent::HistoryTranscriptDelta {
                item_id: stream.item_id,
                delta: stream.delta,
            }))
            .await;
        }
        if effects.items.is_empty() {
            return Ok(());
        }
        self.live_thread_for_persistence("append realtime history")?
            .append_items(
                &effects
                    .items
                    .iter()
                    .cloned()
                    .map(RolloutItem::RealtimeItem)
                    .collect::<Vec<_>>(),
            )
            .await
            .map_err(|error| {
                anyhow::Error::new(error).context(format!(
                    "realtime session {}",
                    effects.items[0].realtime_session_id
                ))
            })?;
        for item in effects.items {
            if !matches!(&item.content, RealtimeItemContent::TranscriptSegment { .. }) {
                self.deliver_event_raw(event(RealtimeEvent::HistoryItemStarted(item.clone())))
                    .await;
            }
            self.deliver_event_raw(event(RealtimeEvent::HistoryItemCompleted(item)))
                .await;
        }
        Ok(())
    }
}
