use crate::{JsonSchema, TS};
use codex_protocol::external_executor::ExternalExecutorEvent;
pub use codex_protocol::external_executor::{
    ExternalExecutorRunParams, ExternalExecutorRunResponse,
};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ExternalExecutorEventParams {
    pub thread_id: String,
    pub turn_id: String,
    pub run_id: String,
    pub event: ExternalExecutorEvent,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export_to = "v2/")]
pub struct ExternalExecutorEventResponse {}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct NativeToolListParams {
    pub thread_id: String,
    pub turn_id: String,
    pub run_id: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export_to = "v2/")]
pub struct NativeToolListResponse {
    pub tools: Vec<serde_json::Value>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct NativeToolCallParams {
    pub thread_id: String,
    pub turn_id: String,
    pub run_id: String,
    pub call_id: String,
    pub namespace: Option<String>,
    pub name: String,
    pub arguments: serde_json::Value,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export_to = "v2/")]
#[serde(rename_all = "camelCase")]
pub struct NativeToolCallResponse {
    pub output: codex_protocol::models::ResponseItem,
    pub is_error: bool,
}

impl NativeToolCallResponse {
    pub fn from_output(output: codex_protocol::models::ResponseItem) -> Self {
        let is_error = match &output {
            codex_protocol::models::ResponseItem::FunctionCallOutput { output, .. }
            | codex_protocol::models::ResponseItem::CustomToolCallOutput { output, .. } => {
                output.success == Some(false)
            }
            _ => false,
        };
        Self { output, is_error }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::models::ResponseItem;
    use serde_json::json;

    #[test]
    fn external_run_requests_preserve_and_require_run_id() {
        fn assert_roundtrip<T: serde::de::DeserializeOwned + Serialize>(wire: serde_json::Value) {
            let request: T = serde_json::from_value(wire.clone()).expect("external run request");
            assert_eq!(serde_json::to_value(request).expect("request wire"), wire);
            let mut missing_run_id = wire;
            missing_run_id.as_object_mut().unwrap().remove("runId");
            assert!(serde_json::from_value::<T>(missing_run_id).is_err());
        }

        assert_roundtrip::<ExternalExecutorRunParams>(json!({
            "threadId": "thread-fixture",
            "turnId": "turn-fixture",
            "runId": "run-fixture",
            "executor": "antigravity",
            "model": "external-test-model",
            "cwd": "/workspace",
            "instructions": "Inspect the task context",
            "effort": null,
            "input": [],
            "fullHistory": false
        }));
        assert_roundtrip::<ExternalExecutorEventParams>(json!({
            "threadId": "thread-fixture",
            "turnId": "turn-fixture",
            "runId": "run-fixture",
            "event": {"type": "textDelta", "delta": "Progress"}
        }));
        assert_roundtrip::<NativeToolListParams>(json!({
            "threadId": "thread-fixture",
            "turnId": "turn-fixture",
            "runId": "run-fixture"
        }));
        assert_roundtrip::<NativeToolCallParams>(json!({
            "threadId": "thread-fixture",
            "turnId": "turn-fixture",
            "runId": "run-fixture",
            "callId": "call-fixture",
            "namespace": null,
            "name": "test_tool",
            "arguments": {"detail": "value"}
        }));
    }

    #[test]
    fn native_tool_call_response_preserves_failure_on_the_wire() {
        for kind in ["function_call_output", "custom_tool_call_output"] {
            for success in [None, Some(true), Some(false)] {
                let mut output: ResponseItem = serde_json::from_value(json!({
                    "type":kind, "call_id":"native-tool", "output":"result"
                }))
                .expect("native output");
                match &mut output {
                    ResponseItem::FunctionCallOutput { output, .. }
                    | ResponseItem::CustomToolCallOutput { output, .. } => output.success = success,
                    _ => unreachable!(),
                }
                let wire = serde_json::to_value(NativeToolCallResponse::from_output(output))
                    .expect("RPC response");
                assert_eq!(wire["isError"], success == Some(false));
                assert_eq!(wire["output"]["output"], "result");
            }
        }
    }
}
