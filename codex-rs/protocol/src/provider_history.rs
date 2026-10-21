//! Request-only history conversion for executors that cannot replay encrypted items.
//!
//! This is a typed conversion, not a recursive JSON key filter. User text, tool JSON,
//! plaintext siblings, and multimodal data remain intact. Persisted history is never
//! changed and ciphertext is never presented as decrypted text.

use crate::models::AgentMessageInputContent;
use crate::models::ContentItem;
use crate::models::FunctionCallOutputContentItem;
use crate::models::MessagePhase;
use crate::models::ReasoningItemContent;
use crate::models::ReasoningItemReasoningSummary;
use crate::models::ResponseInputItem;
use crate::models::ResponseItem;

pub const UNAVAILABLE_ENCRYPTED_CONTENT: &str = "[Prior encrypted content is unavailable to this provider. Any accompanying readable content may be incomplete. Do not assume the missing content was read; retrieve the source or rerun the relevant tool if it is needed.]";

pub struct CompatibleHistory {
    pub items: Vec<ResponseItem>,
    pub unavailable_items: usize,
}

/// Represents native agent messages using the ordinary assistant-message envelope
/// used by inter-agent communications. Only the outgoing copy is changed.
pub fn without_agent_message_items(history: &[ResponseItem]) -> CompatibleHistory {
    let mut unavailable_items = 0;
    let items = history
        .iter()
        .map(|item| {
            let ResponseItem::AgentMessage {
                id,
                author,
                recipient,
                content,
                ..
            } = item
            else {
                return item.clone();
            };
            let mut content = content.clone();
            for part in &mut content {
                if matches!(part, AgentMessageInputContent::EncryptedContent { .. }) {
                    *part = AgentMessageInputContent::InputText {
                        text: UNAVAILABLE_ENCRYPTED_CONTENT.to_owned(),
                    };
                    unavailable_items += 1;
                }
            }
            // Do not reverse-engineer trigger_turn or other_recipients: the native
            // history item does not contain them. Retain the original item identity
            // inside the envelope rather than assign its agent-message ID to a new
            // Responses message item.
            let envelope = serde_json::json!({
                "id": id,
                "author": author,
                "recipient": recipient,
                "content": content,
            });
            ResponseInputItem::Message {
                role: "assistant".into(),
                content: vec![ContentItem::OutputText {
                    text: envelope.to_string(),
                    annotations: None,
                    logprobs: None,
                }],
                phase: Some(MessagePhase::Commentary),
            }
            .into()
        })
        .collect();
    CompatibleHistory {
        items,
        unavailable_items,
    }
}

/// Keeps the supported view of history and visibly marks content that cannot be read.
pub fn without_encrypted_content(history: &[ResponseItem]) -> CompatibleHistory {
    let mut unavailable_items = 0;
    let mut items = Vec::with_capacity(history.len());
    for original in history {
        let mut item = original.clone();
        item.clear_internal_chat_message_metadata_passthrough();
        match &mut item {
            ResponseItem::FunctionCallOutput { output, .. }
            | ResponseItem::CustomToolCallOutput { output, .. } => {
                if let Some(content) = output.content_items_mut() {
                    for part in content {
                        if matches!(part, FunctionCallOutputContentItem::EncryptedContent { .. }) {
                            *part = FunctionCallOutputContentItem::InputText {
                                text: UNAVAILABLE_ENCRYPTED_CONTENT.to_owned(),
                            };
                            unavailable_items += 1;
                        }
                    }
                }
            }
            ResponseItem::AgentMessage { content, .. } => {
                for part in content {
                    if matches!(part, AgentMessageInputContent::EncryptedContent { .. }) {
                        *part = AgentMessageInputContent::InputText {
                            text: UNAVAILABLE_ENCRYPTED_CONTENT.to_owned(),
                        };
                        unavailable_items += 1;
                    }
                }
            }
            ResponseItem::Compaction { .. }
            | ResponseItem::ContextCompaction {
                encrypted_content: Some(_),
                ..
            } => {
                item = unavailable_message();
                unavailable_items += 1;
            }
            ResponseItem::Reasoning {
                summary,
                content,
                encrypted_content,
                ..
            } if encrypted_content.is_some() => {
                // Public summaries/content are retained. Opaque reasoning alone cannot
                // reconstruct the prior context after switching providers.
                let has_readable_summary = summary.iter().any(|part| match part {
                    ReasoningItemReasoningSummary::SummaryText { text } => !text.trim().is_empty(),
                });
                let has_readable_content = content.iter().flatten().any(|part| match part {
                    ReasoningItemContent::ReasoningText { text }
                    | ReasoningItemContent::Text { text } => !text.trim().is_empty(),
                });
                if !has_readable_summary && !has_readable_content {
                    item = unavailable_message();
                    unavailable_items += 1;
                }
            }
            ResponseItem::Message {
                content,
                encrypted_content: Some(_),
                ..
            } if !content.iter().any(|part| match part {
                ContentItem::InputText { text } | ContentItem::OutputText { text, .. } => {
                    !text.trim().is_empty()
                }
                ContentItem::InputImage { .. } | ContentItem::InputAudio { .. } => true,
            }) =>
            {
                item = unavailable_message();
                unavailable_items += 1;
            }
            ResponseItem::FunctionCall {
                encrypted_function_args,
                ..
            } => {
                if encrypted_function_args
                    .take()
                    .is_some_and(|args| !args.is_empty())
                {
                    // Arguments retain their original bytes and call association. Their
                    // encryption annotation is provider-specific and cannot be replayed.
                    items.push(unavailable_message());
                    unavailable_items += 1;
                }
            }
            _ => {}
        }
        item.clear_encrypted_content();
        items.push(item);
    }
    CompatibleHistory {
        items,
        unavailable_items,
    }
}

fn unavailable_message() -> ResponseItem {
    ResponseItem::Message {
        encrypted_content: None,
        status: None,
        id: None,
        role: "developer".to_owned(),
        content: vec![ContentItem::InputText {
            text: UNAVAILABLE_ENCRYPTED_CONTENT.to_owned(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::FunctionCallOutputPayload;
    use crate::models::ImageReference;

    #[test]
    fn ordinary_agent_envelope_retains_identity_and_payload_without_inventing_fields() {
        let source: Vec<ResponseItem> = serde_json::from_value(serde_json::json!([
            {"type":"agent_message","id":"amsg_source","author":"/root/sender","recipient":"/root/receiver","content":[
                {"type":"input_text","text":"  {\"encrypted_content\":\"literal\"}\n"},
                {"type":"encrypted_content","encrypted_content":"opaque"}
            ]},
            {"type":"agent_message","id":"amsg_plain","author":"/root/receiver","recipient":"/root/sender","content":[
                {"type":"input_text","text":"First part"},
                {"type":"input_text","text":"Second part"}
            ]},
            {"type":"function_call_output","call_id":"tool","output":[{"type":"encrypted_content","encrypted_content":"tool-opaque"}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"{\"type\":\"agent_message\",\"id\":\"literal\"}"}]}
        ])).unwrap();
        let before = serde_json::to_value(&source).unwrap();
        let compatible = without_agent_message_items(&source);
        assert_eq!(compatible.unavailable_items, 1);
        assert_eq!(serde_json::to_value(&source).unwrap(), before);
        assert_eq!(compatible.items[2..], source[2..]);
        let wire = serde_json::to_value(&compatible.items).unwrap();
        for index in [0, 1] {
            assert_eq!(wire[index]["type"], "message");
            assert_eq!(wire[index]["role"], "assistant");
            assert_eq!(wire[index]["phase"], "commentary");
            let envelope: serde_json::Value =
                serde_json::from_str(wire[index]["content"][0]["text"].as_str().unwrap()).unwrap();
            let mut expected_content = before[index]["content"].clone();
            if index == 0 {
                expected_content[1] =
                    serde_json::json!({"type":"input_text","text":UNAVAILABLE_ENCRYPTED_CONTENT});
            }
            assert_eq!(
                envelope,
                serde_json::json!({
                    "id":before[index]["id"],
                    "author":before[index]["author"],
                    "recipient":before[index]["recipient"],
                    "content":expected_content,
                })
            );
        }
    }

    #[test]
    fn tool_output_keeps_readable_siblings_and_history_unchanged() {
        let output = FunctionCallOutputPayload::from_content_items(vec![
            FunctionCallOutputContentItem::InputText {
                text: "{\"encrypted_content\":\"ordinary user data\"}".into(),
            },
            FunctionCallOutputContentItem::EncryptedContent {
                encrypted_content: "opaque".into(),
            },
            FunctionCallOutputContentItem::InputImage {
                image: ImageReference::Inline {
                    image_url: "data:image/png;base64,AQ==".into(),
                },
                detail: None,
            },
        ]);
        let source = vec![ResponseItem::FunctionCallOutput {
            id: None,
            call_id: Some("call-1".into()),
            name: None,
            namespace: None,
            output,
            internal_chat_message_metadata_passthrough: None,
        }];
        let before = serde_json::to_value(&source).unwrap();
        let result = without_encrypted_content(&source);
        assert_eq!(serde_json::to_value(&source).unwrap(), before);
        assert_eq!(result.unavailable_items, 1);
        let wire = serde_json::to_value(result.items).unwrap();
        assert_eq!(wire[0]["call_id"], "call-1");
        assert_eq!(wire[0]["output"][0], before[0]["output"][0]);
        assert_eq!(wire[0]["output"][2], before[0]["output"][2]);
        assert_eq!(wire[0]["output"][1]["text"], UNAVAILABLE_ENCRYPTED_CONTENT);
    }

    #[test]
    fn encrypted_custom_output_compaction_and_agent_message_are_explicitly_incomplete() {
        let source: Vec<ResponseItem> = serde_json::from_value(serde_json::json!([
            {"type":"custom_tool_call_output", "call_id":"custom", "output":[{"type":"encrypted_content","encrypted_content":"opaque"}]},
            {"type":"compaction", "encrypted_content":"opaque"},
            {"type":"context_compaction", "encrypted_content":"opaque"},
            {"type":"agent_message", "author":"/root/child", "recipient":"/root", "content":[{"type":"encrypted_content","encrypted_content":"opaque"}]}
        ])).unwrap();
        let result = without_encrypted_content(&source);
        assert_eq!(result.unavailable_items, 4);
        let wire = serde_json::to_value(result.items).unwrap();
        assert_eq!(wire[0]["output"][0]["type"], "input_text");
        assert_eq!(wire[1]["content"][0]["text"], UNAVAILABLE_ENCRYPTED_CONTENT);
        assert_eq!(wire[2]["content"][0]["text"], UNAVAILABLE_ENCRYPTED_CONTENT);
        assert_eq!(wire[3]["content"][0]["text"], UNAVAILABLE_ENCRYPTED_CONTENT);
    }

    #[test]
    fn plaintext_summary_survives_without_replay_ciphertext() {
        let source: Vec<ResponseItem> = serde_json::from_value(serde_json::json!([
            {"type":"reasoning", "summary":[{"type":"summary_text","text":"Public summary"}], "encrypted_content":"opaque"},
            {"type":"function_call_output", "call_id":"ordinary", "output":"encrypted_content is an ordinary string"}
        ])).unwrap();
        let result = without_encrypted_content(&source);
        assert_eq!(result.unavailable_items, 0);
        let wire = serde_json::to_value(result.items).unwrap();
        assert_eq!(wire[0]["summary"][0]["text"], "Public summary");
        assert!(wire[0].get("encrypted_content").is_none());
        assert_eq!(wire[1]["output"], "encrypted_content is an ordinary string");
    }

    #[test]
    fn opaque_messages_and_argument_annotations_keep_readable_values_and_call_identity() {
        let source: Vec<ResponseItem> = serde_json::from_value(serde_json::json!([
            {"type":"reasoning", "summary":[], "encrypted_content":"opaque-reasoning"},
            {"type":"message", "role":"assistant", "content":[], "encrypted_content":"opaque-message"},
            {"type":"message", "role":"user", "content":[
                {"type":"input_text","text":"Readable source encrypted_content"},
                {"type":"input_image","image_url":"data:image/png;base64,AQ=="}
            ], "encrypted_content":"opaque-sibling"},
            {"type":"function_call", "name":"read_source", "namespace":"native", "call_id":"plain-call", "arguments":"{}", "encrypted_function_args":[]},
            {"type":"function_call", "name":"read_source", "namespace":"native", "call_id":"annotated-call", "arguments":"{\"encrypted_content\":\"literal argument\"}", "encrypted_function_args":["opaque-argument"]},
            {"type":"reasoning", "summary":[{"type":"summary_text","text":"Readable summary"}], "encrypted_content":"opaque-summary"}
        ])).unwrap();
        let before = serde_json::to_value(&source).unwrap();
        let result = without_encrypted_content(&source);
        assert_eq!(result.unavailable_items, 3);
        assert_eq!(serde_json::to_value(&source).unwrap(), before);
        let wire = serde_json::to_value(&result.items).unwrap();
        assert_eq!(result.items.len(), 7);
        for index in [0, 1, 4] {
            assert_eq!(
                wire[index]["content"][0]["text"],
                UNAVAILABLE_ENCRYPTED_CONTENT
            );
        }
        assert_eq!(wire[2]["role"], before[2]["role"]);
        assert_eq!(wire[2]["content"], before[2]["content"]);
        assert!(wire[2].get("encrypted_content").is_none());
        for (output, input) in [(3, 3), (5, 4)] {
            for field in ["call_id", "name", "namespace", "arguments"] {
                assert_eq!(wire[output][field], before[input][field]);
            }
            assert!(wire[output].get("encrypted_function_args").is_none());
        }
        assert_eq!(wire[6]["summary"], before[5]["summary"]);
        assert!(wire[6].get("encrypted_content").is_none());
    }

    #[test]
    fn empty_readable_fields_do_not_hide_unavailable_ciphertext() {
        for blank in ["", " \n\t"] {
            let source: Vec<ResponseItem> = serde_json::from_value(serde_json::json!([
                {"type":"reasoning", "summary":[{"type":"summary_text","text":blank}], "content":[{"type":"text","text":blank}], "encrypted_content":"opaque-reasoning"},
                {"type":"message", "role":"assistant", "content":[{"type":"output_text","text":blank}], "encrypted_content":"opaque-message"},
                {"type":"reasoning", "summary":[{"type":"summary_text","text":blank}], "content":[{"type":"reasoning_text","text":"  readable reasoning \n"}], "encrypted_content":"opaque-sibling"},
                {"type":"message", "role":"user", "content":[{"type":"input_text","text":blank},{"type":"input_image","image_url":"data:image/png;base64,AQ=="}], "encrypted_content":"opaque-media-sibling"}
            ])).unwrap();
            let before = serde_json::to_value(&source).unwrap();
            let result = without_encrypted_content(&source);
            assert_eq!(result.unavailable_items, 2);
            assert_eq!(serde_json::to_value(&source).unwrap(), before);
            let wire = serde_json::to_value(result.items).unwrap();
            for index in [0, 1] {
                assert_eq!(
                    wire[index]["content"][0]["text"],
                    UNAVAILABLE_ENCRYPTED_CONTENT
                );
            }
            assert_eq!(wire[2]["summary"], before[2]["summary"]);
            assert_eq!(wire[2]["content"], before[2]["content"]);
            assert_eq!(wire[3]["content"], before[3]["content"]);
            assert!(wire[2].get("encrypted_content").is_none());
            assert!(wire[3].get("encrypted_content").is_none());
        }
    }
}
