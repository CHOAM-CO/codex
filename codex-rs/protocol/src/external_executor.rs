//! Explicit external model execution; tools and thread lifecycle remain owned by Codex.
use crate::models::ResponseItem;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
pub enum ModelExecutor {
    #[default]
    Codex,
    Antigravity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
pub struct ExternalExecutorRunParams {
    pub thread_id: String,
    pub parent_thread_id: Option<String>,
    pub turn_id: String,
    pub run_id: String,
    pub executor: ModelExecutor,
    pub model: String,
    pub cwd: String,
    pub instructions: String,
    pub effort: Option<String>,
    pub input: Vec<ResponseItem>,
    /// Host-authored local context saved without starting an inference turn.
    pub context_updates: Vec<ExternalExecutorContextUpdate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
pub struct ExternalExecutorContextUpdate {
    pub item_id: String,
    pub thread_id: String,
    pub turn_id: String,
    pub order: u64,
    pub item: ResponseItem,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
pub struct ExternalExecutorRunResponse {
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ExternalExecutorEvent {
    TextDelta {
        delta: String,
    },
    /// A call observed from the external engine's own built-in tools.
    Tool {
        #[serde(rename = "callId")]
        call_id: String,
        name: String,
        arguments: serde_json::Value,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present_json"
        )]
        output: Option<serde_json::Value>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present_json"
        )]
        error: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<serde_json::Value>,
    },
}

fn deserialize_present_json<'de, D>(deserializer: D) -> Result<Option<serde_json::Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    serde_json::Value::deserialize(deserializer).map(Some)
}
