//! Defines resume metadata stored directly on a compaction.

use crate::RolloutItem;
use codex_protocol::protocol::MultiAgentVersion;
use codex_protocol::protocol::TurnModelRuntime;
use codex_protocol::turn_input::CyberAccessProgram;
use codex_protocol::turn_input::TurnAttribution;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

/// Resume metadata that is not represented by the companion records after a compaction.
///
/// Presence distinguishes compactions that explicitly persisted these values from older
/// compactions that did not. Each field is authoritative, including an intentionally absent value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CompactionResumeMetadata {
    /// Selected runtime, including when the companion turn context is absent.
    pub multi_agent_version: Option<MultiAgentVersion>,
    /// Turn identity used to admit continuations after cold resume.
    pub last_started_turn_id: Option<String>,
    /// Attribution of the latest regular turn, independent of continuation eligibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_attribution: Option<TurnAttribution>,
    pub previous_turn_settings: Option<PreviousTurnSettings>,
}

/// Previous user-turn settings used to reconstruct context changes after resume.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, JsonSchema)]
pub struct PreviousTurnSettings {
    pub model: String,
    /// Source runtime; absence must not inherit the resumed turn's provider or executor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_runtime: Option<TurnModelRuntime>,
    /// Historical program for this model; absence must not inherit the next turn's selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cyber_access_program: Option<CyberAccessProgram>,
    pub comp_hash: Option<String>,
    pub realtime_active: Option<bool>,
}

/// Returns the runtime version stored in a turn context or compaction resume metadata.
pub fn resume_multi_agent_version(item: &RolloutItem) -> Option<MultiAgentVersion> {
    match item {
        RolloutItem::TurnContext(context) => context.multi_agent_version,
        RolloutItem::Compacted(compacted) => compacted
            .resume_metadata
            .as_ref()
            .and_then(|metadata| metadata.multi_agent_version),
        RolloutItem::SessionMeta(_)
        | RolloutItem::ResponseItem(_)
        | RolloutItem::InterAgentCommunication(_)
        | RolloutItem::InterAgentCommunicationMetadata { .. }
        | RolloutItem::TokenUsageRecord(_)
        | RolloutItem::WorldState(_)
        | RolloutItem::RetainedContext(_)
        | RolloutItem::SecurityRiskScore(_)
        | RolloutItem::RealtimeItem(_)
        | RolloutItem::EventMsg(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::external_executor::ModelExecutor;
    use serde_json::json;

    #[test]
    fn previous_turn_settings_preserve_optional_model_runtime() {
        let legacy = json!({"model": "source-model"});
        let mut settings: PreviousTurnSettings = serde_json::from_value(legacy).unwrap();
        assert_eq!(settings.model_runtime, None);
        assert!(
            serde_json::to_value(&settings)
                .unwrap()
                .get("model_runtime")
                .is_none()
        );

        settings.model_runtime = Some(TurnModelRuntime {
            model_provider: "source-provider".to_owned(),
            model_executor: ModelExecutor::Codex,
            model_context_window: Some(256_000),
        });
        let persisted = serde_json::to_value(&settings).unwrap();
        assert_eq!(
            persisted["model_runtime"],
            json!({
                "model_provider": "source-provider",
                "model_executor": "codex",
                "model_context_window": 256_000
            })
        );
        assert_eq!(
            serde_json::from_value::<PreviousTurnSettings>(persisted).unwrap(),
            settings
        );
    }
}
