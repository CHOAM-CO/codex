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
    pub turn_id: String,
    pub run_id: String,
    pub executor: ModelExecutor,
    pub model: String,
    pub cwd: String,
    pub instructions: String,
    pub effort: Option<String>,
    pub input: Vec<ResponseItem>,
    pub full_history: bool,
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
    /// A completed call executed by the external engine's own built-in tools.
    Tool {
        #[serde(rename = "callId")]
        call_id: String,
        name: String,
        arguments: serde_json::Value,
        output: serde_json::Value,
    },
}
