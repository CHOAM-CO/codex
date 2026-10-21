//! Client-owned model transport using Codex-owned turns, history, and tool handlers.
use crate::CodexThread;
use crate::session::TurnInput;
use crate::session::session::Session;
use crate::session::step_context::StepContext;
use crate::session::turn::run_hooks_and_record_inputs;
use crate::session::turn_context::TurnContext;
use crate::tools::parallel::ToolCallRuntime;
use crate::turn_diff_tracker::TurnDiffTracker;
use codex_protocol::error::{CodexErr, Result};
use codex_protocol::external_executor::{
    ExternalExecutorEvent, ExternalExecutorRunParams, ExternalExecutorRunResponse,
};
use codex_protocol::items::{AgentMessageContent, AgentMessageItem, TurnItem};
use codex_protocol::models::{ContentItem, FunctionCallOutputPayload, MessagePhase, ResponseItem};
use codex_protocol::protocol::{AgentMessageContentDeltaEvent, EventMsg};
use codex_thread_store::PersistContext;
use codex_tools::ToolName;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

pub(crate) enum ExternalMessage {
    Event(ExternalExecutorEvent),
    Complete(ExternalExecutorRunResponse),
}

fn external_tool_output_view(output: &ResponseItem) -> ResponseItem {
    codex_protocol::provider_history::without_encrypted_content(std::slice::from_ref(output))
        .items
        .pop()
        .expect("compatible history preserves every tool output")
}

fn function_call(
    call_id: String,
    namespace: Option<String>,
    name: String,
    arguments: String,
) -> ResponseItem {
    ResponseItem::FunctionCall {
        id: None,
        encrypted_content: None,
        status: None,
        namespace,
        name,
        arguments,
        encrypted_function_args: None,
        call_id,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn message_item(id: &str, text: &str, phase: MessagePhase) -> TurnItem {
    TurnItem::AgentMessage(AgentMessageItem {
        id: id.to_string(),
        content: vec![AgentMessageContent::Text {
            text: text.to_string(),
        }],
        phase: Some(phase),
        memory_citation: None,
        delivery: None,
        questions: None,
    })
}

async fn record_message(
    session: &Session,
    step: &StepContext,
    id: &str,
    text: String,
    completed: bool,
) {
    let item = ResponseItem::Message {
        id: Some(codex_protocol::ResponseItemId::from_server(id.to_string())),
        encrypted_content: None,
        status: (!completed).then(|| "incomplete".to_string()),
        role: "assistant".into(),
        content: vec![ContentItem::OutputText {
            text,
            annotations: None,
            logprobs: None,
        }],
        phase: Some(MessagePhase::FinalAnswer),
        internal_chat_message_metadata_passthrough: None,
    };
    if completed {
        crate::stream_events_utils::record_completed_response_item(session, step, &item).await;
    } else {
        session
            .record_conversation_items(&step.turn, &step.settings.model_info, &[item])
            .await;
    }
}

pub(crate) async fn run_external_turn(
    session: Arc<Session>,
    turn: Arc<TurnContext>,
    input: Vec<TurnInput>,
    cancel: CancellationToken,
) -> Result<Option<String>> {
    let step = session
        .capture_step_context(Arc::clone(&turn), &cancel)
        .await?;
    session
        .record_context_updates_and_set_reference_context_item(&step)
        .await?;
    if run_hooks_and_record_inputs(
        &session,
        &turn,
        &step.settings.model_info,
        &input,
        PersistContext::Standard,
    )
    .await
    {
        return Ok(None);
    }
    let runtime = ToolCallRuntime::new(
        Arc::clone(&session),
        Arc::clone(&step),
        Arc::new(Mutex::new(TurnDiffTracker::new())),
    );
    let id = codex_protocol::ResponseItemId::new("msg").to_string();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let request_cancel = cancel.child_token();
    let _cancel_on_finish = request_cancel.clone().drop_guard();
    {
        let active = session.active_turn.lock().await;
        let active = active.as_ref().ok_or(CodexErr::TurnAborted)?;
        let mut state = active.turn_state.lock().await;
        state.external_run_id = Some(id.clone());
        state.external_executor = Some(tx);
        state.external_tool_runtime = Some(runtime);
        state.external_cancellation = Some(request_cancel);
    }
    let history = session.conversation_history_snapshot().await;
    session
        .send_event(
            &turn,
            EventMsg::ExternalExecutorRequest(ExternalExecutorRunParams {
                thread_id: session.thread_id.to_string(),
                turn_id: turn.sub_id.clone(),
                run_id: id.clone(),
                executor: turn.config.model_executor,
                model: step.settings.model_info.slug.clone(),
                cwd: turn.config.cwd.to_string_lossy().into_owned(),
                full_history: true,
                input: codex_protocol::provider_history::without_encrypted_content(
                    &history.items().cloned().collect::<Vec<_>>(),
                )
                .items,
                instructions: session.get_prompt_base_instructions().await.text,
                effort: step
                    .settings
                    .effective_reasoning_effort()
                    .map(|effort| effort.to_string()),
            }),
        )
        .await;
    let mut text = String::new();
    let mut started = false;
    loop {
        let next = tokio::select! {
            _ = cancel.cancelled() => {
                if started { record_message(&session, &step, &id, text, false).await; }
                return Err(CodexErr::TurnAborted);
            },
            next = rx.recv() => next.ok_or_else(|| CodexErr::Fatal("external executor disconnected".into()))?,
        };
        match next {
            ExternalMessage::Event(ExternalExecutorEvent::TextDelta { delta }) => {
                if !started {
                    session
                        .emit_turn_item_started(
                            &turn,
                            &message_item(&id, "", MessagePhase::FinalAnswer),
                        )
                        .await;
                    started = true;
                }
                text.push_str(&delta);
                session
                    .send_event(
                        &turn,
                        EventMsg::AgentMessageContentDelta(AgentMessageContentDeltaEvent {
                            thread_id: session.thread_id.to_string(),
                            turn_id: turn.sub_id.clone(),
                            item_id: id.clone(),
                            delta,
                        }),
                    )
                    .await;
            }
            ExternalMessage::Event(ExternalExecutorEvent::Tool {
                call_id,
                name,
                arguments,
                output,
            }) => {
                let call = function_call(
                    call_id.clone(),
                    Some("antigravity".into()),
                    name.clone(),
                    arguments.to_string(),
                );
                let result = ResponseItem::FunctionCallOutput {
                    id: None,
                    call_id: Some(call_id),
                    name: Some(name),
                    namespace: Some("antigravity".into()),
                    output: FunctionCallOutputPayload::from_text(output.to_string()),
                    internal_chat_message_metadata_passthrough: None,
                };
                session
                    .record_conversation_items(&turn, &step.settings.model_info, &[call, result])
                    .await;
            }
            ExternalMessage::Complete(result) => {
                if started {
                    record_message(&session, &step, &id, text.clone(), result.success).await;
                    session
                        .emit_turn_item_completed(
                            &turn,
                            message_item(&id, &text, MessagePhase::FinalAnswer),
                        )
                        .await;
                }
                if !result.success {
                    return Err(CodexErr::Fatal(
                        result
                            .error
                            .unwrap_or_else(|| "external executor failed".into()),
                    ));
                }
                return Ok(started.then_some(text));
            }
        }
    }
}

impl Session {
    async fn external_context(
        &self,
        turn_id: &str,
        run_id: &str,
    ) -> Result<(Arc<StepContext>, ToolCallRuntime, CancellationToken)> {
        let active = self.active_turn.lock().await;
        let active = active
            .as_ref()
            .ok_or_else(|| CodexErr::InvalidRequest("thread has no active turn".into()))?;
        let task = active.task.as_ref().ok_or(CodexErr::TurnAborted)?;
        if task.turn_context.sub_id != turn_id || task.cancellation_token.is_cancelled() {
            return Err(CodexErr::InvalidRequest(
                "external turn is no longer active".into(),
            ));
        }
        let state = active.turn_state.lock().await;
        if state.external_run_id.as_deref() != Some(run_id) {
            return Err(CodexErr::InvalidRequest(
                "external run is no longer active".into(),
            ));
        }
        let step = state
            .last_known_step_context
            .clone()
            .ok_or_else(|| CodexErr::InvalidRequest("external tools are not ready".into()))?;
        let runtime = state.external_tool_runtime.clone().ok_or_else(|| {
            CodexErr::InvalidRequest("turn does not use an external executor".into())
        })?;
        let cancellation = state
            .external_cancellation
            .as_ref()
            .filter(|token| !token.is_cancelled())
            .ok_or(CodexErr::TurnAborted)?
            .child_token();
        Ok((step, runtime, cancellation))
    }

    pub(crate) async fn external_tool_list(
        &self,
        turn_id: &str,
        run_id: &str,
    ) -> Result<Vec<serde_json::Value>> {
        let (step, _, _) = self.external_context(turn_id, run_id).await?;
        step.tool_router
            .external_tool_specs()
            .into_iter()
            .map(|spec| serde_json::to_value(spec).map_err(|err| CodexErr::Fatal(err.to_string())))
            .collect()
    }

    pub(crate) async fn external_tool_call(
        &self,
        turn_id: &str,
        run_id: &str,
        call_id: String,
        namespace: Option<String>,
        name: String,
        arguments: serde_json::Value,
    ) -> Result<ResponseItem> {
        let (step, runtime, cancel) = self.external_context(turn_id, run_id).await?;
        let tool_name = ToolName::new(namespace.clone(), name.clone());
        if !step.tool_router.exposes_tool(&tool_name) {
            return Err(CodexErr::InvalidRequest(format!(
                "tool {tool_name} is not exposed in this turn"
            )));
        }
        let freeform = step.tool_router.registered_tool(&tool_name).map(|entry| entry.spec()).is_some_and(|spec| match spec.as_ref() {
            codex_tools::ToolSpec::Freeform(_) => true,
            codex_tools::ToolSpec::Namespace(namespace) => namespace.tools.iter().any(|tool| matches!(tool, codex_tools::ResponsesApiNamespaceTool::Custom(custom) if custom.name == name)),
            _ => false,
        });
        let call = if freeform {
            let input = arguments
                .as_str()
                .ok_or_else(|| {
                    CodexErr::InvalidRequest("freeform tool arguments must be a string".into())
                })?
                .to_string();
            ResponseItem::CustomToolCall {
                id: None,
                encrypted_content: None,
                status: None,
                call_id: call_id.clone(),
                namespace,
                name,
                input,
                internal_chat_message_metadata_passthrough: None,
            }
        } else {
            function_call(call_id.clone(), namespace, name, arguments.to_string())
        };
        let dispatch = crate::tools::router::ToolRouter::build_tool_call(call.clone())
            .map_err(|err| CodexErr::InvalidRequest(err.to_string()))?
            .ok_or_else(|| {
                CodexErr::InvalidRequest("item is not an executable tool call".into())
            })?;
        self.record_conversation_items(&step.turn, &step.settings.model_info, &[call])
            .await;
        let result = runtime.handle_external_tool_call(dispatch, cancel).await?;
        crate::stream_events_utils::mark_thread_memory_mode_polluted_if_external_context(
            self,
            &step.turn,
            &result.item,
        )
        .await;
        let output = external_tool_output_view(&result.item);
        self.record_annotated_conversation_items(
            &step.turn,
            &step.settings.model_info,
            vec![result],
        )
        .await;
        Ok(output)
    }

    pub(crate) async fn external_executor_event(
        &self,
        turn_id: &str,
        run_id: &str,
        event: ExternalExecutorEvent,
    ) -> Result<()> {
        self.send_external(turn_id, run_id, ExternalMessage::Event(event))
            .await
    }

    pub(crate) async fn external_executor_complete(
        &self,
        turn_id: &str,
        run_id: &str,
        result: ExternalExecutorRunResponse,
    ) -> Result<()> {
        self.send_external(turn_id, run_id, ExternalMessage::Complete(result))
            .await
    }

    async fn send_external(
        &self,
        turn_id: &str,
        run_id: &str,
        event: ExternalMessage,
    ) -> Result<()> {
        let active = self.active_turn.lock().await;
        let active = active.as_ref().ok_or(CodexErr::TurnAborted)?;
        let task = active.task.as_ref().ok_or(CodexErr::TurnAborted)?;
        if task.turn_context.sub_id != turn_id || task.cancellation_token.is_cancelled() {
            return Err(CodexErr::TurnAborted);
        }
        let state = active.turn_state.lock().await;
        if state.external_run_id.as_deref() != Some(run_id) {
            return Err(CodexErr::InvalidRequest(
                "external run is no longer active".into(),
            ));
        }
        state
            .external_executor
            .as_ref()
            .ok_or_else(|| {
                CodexErr::InvalidRequest("turn does not use an external executor".into())
            })?
            .send(event)
            .map_err(|_| CodexErr::TurnAborted)
    }
}

impl CodexThread {
    pub async fn external_tool_list(
        &self,
        turn_id: &str,
        run_id: &str,
    ) -> Result<Vec<serde_json::Value>> {
        self.session.external_tool_list(turn_id, run_id).await
    }
    pub async fn external_tool_call(
        &self,
        turn_id: &str,
        run_id: &str,
        call_id: String,
        namespace: Option<String>,
        name: String,
        arguments: serde_json::Value,
    ) -> Result<ResponseItem> {
        self.session
            .external_tool_call(turn_id, run_id, call_id, namespace, name, arguments)
            .await
    }
    pub async fn external_executor_event(
        &self,
        turn_id: &str,
        run_id: &str,
        event: ExternalExecutorEvent,
    ) -> Result<()> {
        self.session
            .external_executor_event(turn_id, run_id, event)
            .await
    }
    pub async fn external_executor_complete(
        &self,
        turn_id: &str,
        run_id: &str,
        result: ExternalExecutorRunResponse,
    ) -> Result<()> {
        self.session
            .external_executor_complete(turn_id, run_id, result)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::external_executor::ModelExecutor;
    use codex_protocol::protocol::TurnAbortReason;
    use serde_json::json;
    use std::time::Duration;

    #[test]
    fn native_tool_return_converts_ciphertext_without_mutating_native_output() {
        let source: ResponseItem = serde_json::from_value(json!({
            "type":"function_call_output", "call_id":"native-call", "output":[
                {"type":"input_text", "text":"readable result"},
                {"type":"encrypted_content", "encrypted_content":"opaque native output"},
                {"type":"input_image", "image_url":"data:image/png;base64,AQ=="}
            ]
        }))
        .expect("typed native output");
        let before = source.clone();
        let result = external_tool_output_view(&source);
        assert_eq!(source, before);
        let wire = serde_json::to_value(result).expect("external wire output");
        assert_eq!(wire["call_id"], "native-call");
        assert_eq!(wire["output"][0]["text"], "readable result");
        assert_eq!(
            wire["output"][1]["text"],
            codex_protocol::provider_history::UNAVAILABLE_ENCRYPTED_CONTENT
        );
        assert_eq!(wire["output"][2]["image_url"], "data:image/png;base64,AQ==");
    }

    fn run_test<F, Fut>(test: F)
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        core_test_support::test_codex::run_test_with_large_stack(
            "external-executor",
            move || async move {
                test().await;
                Ok(())
            },
        )
        .expect("external executor behavior");
    }

    async fn next_request(
        rx: &async_channel::Receiver<codex_protocol::protocol::Event>,
    ) -> ExternalExecutorRunParams {
        loop {
            let event = tokio::time::timeout(Duration::from_secs(10), rx.recv())
                .await
                .expect("external request arrives")
                .expect("event channel open");
            if let EventMsg::ExternalExecutorRequest(request) = event.msg {
                return request;
            }
        }
    }

    async fn complete(session: &Session, turn: &TurnContext, run_id: &str) {
        session
            .external_executor_complete(
                &turn.sub_id,
                &run_id,
                ExternalExecutorRunResponse {
                    success: true,
                    error: None,
                },
            )
            .await
            .expect("completion accepted");
    }

    fn output_text(output: ResponseItem) -> String {
        let ResponseItem::FunctionCallOutput { output, .. } = output else {
            panic!("native function output expected");
        };
        assert_ne!(output.success, Some(false), "{output:?}");
        output
            .text_content()
            .expect("native text output")
            .to_owned()
    }

    fn output_json(output: ResponseItem) -> serde_json::Value {
        let text = output_text(output);
        serde_json::from_str(&text)
            .unwrap_or_else(|error| panic!("native JSON output: {error}; received {text:?}"))
    }

    async fn setup() -> (
        Arc<Session>,
        Arc<TurnContext>,
        async_channel::Receiver<codex_protocol::protocol::Event>,
        String,
    ) {
        let (session, mut turn, rx) =
            crate::session::tests::make_session_and_context_with_rx().await;
        let config = Arc::make_mut(&mut Arc::get_mut(&mut turn).expect("exclusive turn").config);
        config.model_executor = ModelExecutor::Antigravity;
        config.update_plan_enabled = true;
        session
            .spawn_task(
                Arc::clone(&turn),
                Vec::new(),
                crate::tasks::RegularTask::new(),
            )
            .await;
        let request = next_request(&rx).await;
        assert_eq!(request.executor, ModelExecutor::Antigravity);
        assert!(request.full_history);
        assert_eq!(request.turn_id, turn.sub_id);
        (session, turn, rx, request.run_id)
    }

    #[test]
    fn external_text_becomes_native_message_history_and_completion() {
        run_test(|| async {
            let (session, turn, rx, run_id) = setup().await;
            session
                .external_executor_event(
                    &turn.sub_id,
                    &run_id,
                    ExternalExecutorEvent::TextDelta {
                        delta: "hello ".into(),
                    },
                )
                .await
                .expect("delta accepted");
            session
                .external_executor_event(
                    &turn.sub_id,
                    &run_id,
                    ExternalExecutorEvent::TextDelta {
                        delta: "world".into(),
                    },
                )
                .await
                .expect("delta accepted");
            session
                .external_executor_complete(
                    &turn.sub_id,
                    &run_id,
                    ExternalExecutorRunResponse {
                        success: true,
                        error: None,
                    },
                )
                .await
                .expect("completion accepted");
            let mut deltas = String::new();
            loop {
                let event = tokio::time::timeout(Duration::from_secs(10), rx.recv())
                    .await
                    .expect("completion arrives")
                    .expect("event channel open");
                match event.msg {
                    EventMsg::AgentMessageContentDelta(event) => deltas.push_str(&event.delta),
                    EventMsg::TurnComplete(_) => break,
                    _ => (),
                }
            }
            assert_eq!(deltas, "hello world");
            let history = session.conversation_history_snapshot().await;
            assert!(history.items().any(|item| matches!(item, ResponseItem::Message { role, content, .. } if role == "assistant" && content.iter().any(|item| matches!(item, ContentItem::OutputText { text, .. } if text == "hello world")))));
            assert!(
                session
                    .external_tool_list(&turn.sub_id, &run_id)
                    .await
                    .is_err()
            );
        });
    }

    #[test]
    fn native_plan_tool_runs_in_external_turn_without_model_response() {
        run_test(|| async {
            let (session, turn, rx, run_id) = setup().await;
            let tools = session
                .external_tool_list(&turn.sub_id, &run_id)
                .await
                .expect("native catalog");
            assert!(!tools.is_empty());
            let result = session
                .external_tool_call(
                    &turn.sub_id,
                    &run_id,
                    "plan-call".into(),
                    None,
                    "update_plan".into(),
                    json!({"plan":[{"step":"Inspect input","status":"in_progress"}]}),
                )
                .await
                .expect("native plan handler");
            assert!(matches!(result, ResponseItem::FunctionCallOutput { .. }));
            let mut saw_plan = false;
            while let Ok(event) = rx.try_recv() {
                if matches!(event.msg, EventMsg::PlanUpdate(_)) {
                    saw_plan = true;
                }
            }
            assert!(saw_plan);
            let history = session.conversation_history_snapshot().await;
            assert!(history.items().any(|item| matches!(item, ResponseItem::FunctionCall { call_id, .. } if call_id == "plan-call")));
            assert!(history.items().any(|item| matches!(item, ResponseItem::FunctionCallOutput { call_id: Some(call_id), .. } if call_id == "plan-call")));
            session
                .external_executor_complete(
                    &turn.sub_id,
                    &run_id,
                    ExternalExecutorRunResponse {
                        success: true,
                        error: None,
                    },
                )
                .await
                .expect("complete");
        });
    }

    #[test]
    fn interrupted_external_turn_rejects_late_events_and_tool_calls() {
        run_test(|| async {
            let (session, turn, _rx, run_id) = setup().await;
            session.abort_all_tasks(TurnAbortReason::Interrupted).await;
            assert!(
                session
                    .external_executor_event(
                        &turn.sub_id,
                        &run_id,
                        ExternalExecutorEvent::TextDelta {
                            delta: "late".into()
                        }
                    )
                    .await
                    .is_err()
            );
            assert!(
                session
                    .external_tool_call(
                        &turn.sub_id,
                        &run_id,
                        "late-call".into(),
                        None,
                        "update_plan".into(),
                        json!({"plan":[]})
                    )
                    .await
                    .is_err()
            );
        });
    }

    #[test]
    fn queued_steering_is_preserved_in_next_external_request() {
        run_test(|| async {
            let (session, turn, rx, first_run_id) = setup().await;
            let (other, other_turn, _other_rx, other_run_id) = setup().await;
            assert_ne!(session.thread_id, other.thread_id);
            let other_history = other
                .clone_history()
                .await
                .raw_items()
                .cloned()
                .collect::<Vec<_>>();
            let input = ResponseItem::Message {
                id: None,
                encrypted_content: None,
                status: None,
                role: "user".into(),
                content: vec![ContentItem::InputText {
                    text: "include the new requirement".into(),
                }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            };
            let state = session
                .active_turn
                .lock()
                .await
                .as_ref()
                .expect("active external turn")
                .turn_state
                .clone();
            session
                .input_queue
                .extend_pending_input_for_turn_state(
                    &state,
                    vec![TurnInput::ResponseItem(input.clone().into())],
                )
                .await;
            complete(&session, &turn, &first_run_id).await;
            let request = next_request(&rx).await;
            assert_ne!(request.run_id, first_run_id);
            let run_id = request.run_id.clone();
            assert_eq!(request.thread_id, session.thread_id.to_string());
            assert_eq!(request.turn_id, turn.sub_id);
            assert_eq!(
                request.input.iter().filter(|item| matches!(item,
                    ResponseItem::Message { role, content, .. } if role == "user" && content.iter().any(|part|
                    matches!(part, ContentItem::InputText { text } if text == "include the new requirement"))
                )).count(),
                1
            );
            assert!(
                !session
                    .input_queue
                    .has_pending_input(&session.active_turn)
                    .await
            );
            let current_history = session
                .clone_history()
                .await
                .raw_items()
                .cloned()
                .collect::<Vec<_>>();
            assert!(
                session
                    .external_tool_list(&turn.sub_id, &first_run_id)
                    .await
                    .is_err()
            );
            assert!(
                session
                    .external_tool_call(
                        &turn.sub_id,
                        &first_run_id,
                        "stale-native-call".into(),
                        None,
                        "update_plan".into(),
                        json!({"plan":[{"step":"stale plan", "status":"in_progress"}]}),
                    )
                    .await
                    .is_err()
            );
            for event in [
                ExternalExecutorEvent::TextDelta {
                    delta: "stale text".into(),
                },
                ExternalExecutorEvent::Tool {
                    call_id: "stale-built-in-call".into(),
                    name: "read_file".into(),
                    arguments: json!({"path":"stale.md"}),
                    output: json!({"text":"stale result"}),
                },
            ] {
                assert!(
                    session
                        .external_executor_event(&turn.sub_id, &first_run_id, event)
                        .await
                        .is_err()
                );
            }
            assert!(
                session
                    .external_executor_complete(
                        &turn.sub_id,
                        &first_run_id,
                        ExternalExecutorRunResponse {
                            success: true,
                            error: None
                        },
                    )
                    .await
                    .is_err()
            );
            assert!(
                other
                    .external_tool_list(&other_turn.sub_id, &run_id)
                    .await
                    .is_err()
            );
            assert_eq!(
                session
                    .clone_history()
                    .await
                    .raw_items()
                    .cloned()
                    .collect::<Vec<_>>(),
                current_history
            );
            assert!(
                !session
                    .external_tool_list(&turn.sub_id, &run_id)
                    .await
                    .expect("current catalog")
                    .is_empty()
            );
            let result = session
                .external_tool_call(
                    &turn.sub_id,
                    &run_id,
                    "current-native-call".into(),
                    None,
                    "update_plan".into(),
                    json!({"plan":[{"step":"current plan", "status":"in_progress"}]}),
                )
                .await
                .expect("current native tool");
            assert!(
                matches!(result, ResponseItem::FunctionCallOutput { call_id: Some(ref call_id), .. }
                if call_id == "current-native-call")
            );
            let arguments =
                json!({"path":"current.md", "literal":{"encrypted_content":"user text"}});
            let output = json!({"text":"current file content", "lines":2});
            session
                .external_executor_event(
                    &turn.sub_id,
                    &run_id,
                    ExternalExecutorEvent::Tool {
                        call_id: "current-built-in-call".into(),
                        name: "read_file".into(),
                        arguments: arguments.clone(),
                        output: output.clone(),
                    },
                )
                .await
                .expect("current built-in event");
            session
                .external_executor_event(
                    &turn.sub_id,
                    &run_id,
                    ExternalExecutorEvent::TextDelta {
                        delta: "current response".into(),
                    },
                )
                .await
                .expect("current text event");
            complete(&session, &turn, &run_id).await;
            let mut deltas = Vec::new();
            let mut plans = 0;
            loop {
                let event = tokio::time::timeout(Duration::from_secs(10), rx.recv())
                    .await
                    .expect("current run completion")
                    .expect("events open");
                match event.msg {
                    EventMsg::AgentMessageContentDelta(delta) => {
                        assert_eq!(delta.item_id, run_id);
                        deltas.push(delta.delta);
                    }
                    EventMsg::PlanUpdate(_) => plans += 1,
                    EventMsg::TurnComplete(_) => break,
                    _ => (),
                }
            }
            assert_eq!(deltas, ["current response"]);
            assert_eq!(plans, 1);
            let history = session.clone_history().await;
            let mut native_pair = history.raw_items().filter(|item| matches!(item,
                ResponseItem::FunctionCall { call_id, .. } | ResponseItem::FunctionCallOutput { call_id: Some(call_id), .. }
                if call_id == "current-native-call"
            )).cloned().collect::<Vec<_>>();
            assert_eq!(native_pair.len(), 2);
            assert!(
                matches!(native_pair.remove(0), ResponseItem::FunctionCall { namespace: None, ref name, .. } if name == "update_plan")
            );
            assert!(matches!(
                native_pair.remove(0),
                ResponseItem::FunctionCallOutput { .. }
            ));
            let mut built_in_pair = history.raw_items().filter(|item| matches!(item,
                ResponseItem::FunctionCall { call_id, .. } | ResponseItem::FunctionCallOutput { call_id: Some(call_id), .. }
                if call_id == "current-built-in-call"
            )).cloned().collect::<Vec<_>>();
            for item in &mut built_in_pair {
                item.clear_internal_chat_message_metadata_passthrough();
            }
            let mut expected_built_in_pair = vec![
                function_call(
                    "current-built-in-call".into(),
                    Some("antigravity".into()),
                    "read_file".into(),
                    arguments.to_string(),
                ),
                ResponseItem::FunctionCallOutput {
                    id: None,
                    call_id: Some("current-built-in-call".into()),
                    name: Some("read_file".into()),
                    namespace: Some("antigravity".into()),
                    output: FunctionCallOutputPayload::from_text(output.to_string()),
                    internal_chat_message_metadata_passthrough: None,
                },
            ];
            assert_eq!(built_in_pair.len(), 2);
            assert_ne!(built_in_pair[0].id(), built_in_pair[1].id());
            for (expected, recorded) in expected_built_in_pair.iter_mut().zip(&built_in_pair) {
                assert!(recorded.id().is_some_and(|id| !id.is_empty()));
                expected.set_id(recorded.id().cloned());
            }
            assert_eq!(built_in_pair, expected_built_in_pair);
            assert_eq!(
                other
                    .clone_history()
                    .await
                    .raw_items()
                    .cloned()
                    .collect::<Vec<_>>(),
                other_history
            );
            assert!(
                !other
                    .external_tool_list(&other_turn.sub_id, &other_run_id)
                    .await
                    .expect("other run remains active")
                    .is_empty()
            );
            other.abort_all_tasks(TurnAbortReason::Interrupted).await;
        });
    }

    #[test]
    fn previous_turn_events_cannot_mutate_new_external_turn() {
        run_test(|| async {
            let (session, first, rx, first_run_id) = setup().await;
            session.abort_all_tasks(TurnAbortReason::Interrupted).await;
            let mut second = session.new_default_turn().await;
            Arc::make_mut(&mut Arc::get_mut(&mut second).expect("exclusive turn").config)
                .model_executor = ModelExecutor::Antigravity;
            assert_ne!(first.sub_id, second.sub_id);
            session
                .spawn_task(
                    Arc::clone(&second),
                    Vec::new(),
                    crate::tasks::RegularTask::new(),
                )
                .await;
            let second_request = next_request(&rx).await;
            assert_eq!(second_request.turn_id, second.sub_id);
            let second_run_id = second_request.run_id;
            assert!(
                session
                    .external_executor_complete(
                        &first.sub_id,
                        &first_run_id,
                        ExternalExecutorRunResponse {
                            success: true,
                            error: None
                        },
                    )
                    .await
                    .is_err()
            );
            assert!(
                session
                    .external_executor_event(
                        &first.sub_id,
                        &first_run_id,
                        ExternalExecutorEvent::TextDelta {
                            delta: "stale text".into()
                        },
                    )
                    .await
                    .is_err()
            );
            session
                .external_executor_event(
                    &second.sub_id,
                    &second_run_id,
                    ExternalExecutorEvent::TextDelta {
                        delta: "current text".into(),
                    },
                )
                .await
                .expect("current event accepted");
            complete(&session, &second, &second_run_id).await;
            let mut deltas = Vec::new();
            loop {
                let event = tokio::time::timeout(Duration::from_secs(10), rx.recv())
                    .await
                    .expect("completion arrives")
                    .expect("event stream open");
                match event.msg {
                    EventMsg::AgentMessageContentDelta(delta) => {
                        assert_eq!(delta.turn_id, second.sub_id);
                        deltas.push(delta.delta);
                    }
                    EventMsg::TurnComplete(_) => break,
                    _ => (),
                }
            }
            assert_eq!(deltas, vec!["current text"]);
        });
    }

    #[test]
    fn external_request_completion_cancels_pending_native_tool() {
        run_test(|| async {
            let (session, turn, rx) =
                crate::session::tests::make_session_and_context_with_auth_and_config_and_rx(
                    codex_login::CodexAuth::from_api_key("test"),
                    Vec::new(),
                    |config| {
                        config.model_executor = ModelExecutor::Antigravity;
                        config
                            .features
                            .enable(codex_features::Feature::MultiAgentV2)
                            .expect("enable v2");
                    },
                )
                .await;
            session
                .spawn_task(
                    Arc::clone(&turn),
                    Vec::new(),
                    crate::tasks::RegularTask::new(),
                )
                .await;
            let run_id = next_request(&rx).await.run_id;
            let call = tokio::spawn({
                let session = Arc::clone(&session);
                let turn = Arc::clone(&turn);
                let run_id = run_id.clone();
                async move {
                    session
                        .external_tool_call(
                            &turn.sub_id,
                            &run_id,
                            "pending-wait".into(),
                            turn.config.multi_agent_v2.tool_namespace.clone(),
                            "wait_agent".into(),
                            json!({"timeout_ms":10_000}),
                        )
                        .await
                }
            });
            loop {
                let event = tokio::time::timeout(Duration::from_secs(10), rx.recv())
                    .await
                    .expect("native tool started")
                    .expect("events open");
                if matches!(event.msg, EventMsg::ItemStarted(codex_protocol::protocol::ItemStartedEvent {
                item: TurnItem::CollabAgentToolCall(ref item), ..
            }) if item.id == "pending-wait")
                {
                    break;
                }
            }
            complete(&session, &turn, &run_id).await;
            let result = tokio::time::timeout(Duration::from_secs(2), call)
                .await
                .expect("native tool cancelled before its timeout")
                .expect("call task joined")
                .expect("native cancellation output");
            let ResponseItem::FunctionCallOutput { output, .. } = result else {
                panic!("native tool output expected");
            };
            assert!(
                output
                    .body
                    .to_text()
                    .expect("abort text")
                    .starts_with("aborted by user")
            );
        });
    }

    #[test]
    fn native_child_tools_share_external_turn_context_and_plaintext_messages() {
        run_test(|| async {
            use codex_features::Feature;
            use codex_protocol::AgentPath;
            use codex_protocol::protocol::{InterAgentCommunication, Op};

            let (mut session, turn, rx) =
                crate::session::tests::make_session_and_context_with_auth_and_config_and_rx(
                    codex_login::CodexAuth::from_api_key("test"),
                    Vec::new(),
                    |config| {
                        config.model_executor = ModelExecutor::Antigravity;
                        config.model = Some("external-test-model".into());
                        config.multi_agent_v2.tool_namespace = Some("native_agents".into());
                        config
                            .features
                            .enable(Feature::MultiAgentV2)
                            .expect("enable v2");
                    },
                )
                .await;
            let manager = crate::ThreadManager::with_models_provider_for_tests(
                codex_login::CodexAuth::from_api_key("test"),
                codex_model_provider_info::built_in_model_providers(None)["openai"].clone(),
            );
            let root = manager
                .start_thread(crate::thread_manager::StartThreadOptions::new(
                    (*turn.config).clone(),
                ))
                .await
                .expect("native root");
            let owned = Arc::get_mut(&mut session).expect("exclusive session");
            let control = manager.agent_control();
            owned.services.local_agent_runtime = control.runtime.clone();
            owned.services.agent_control = Arc::new(control);
            owned.thread_id = root.thread_id;
            session
                .spawn_task(
                    Arc::clone(&turn),
                    Vec::new(),
                    crate::tasks::RegularTask::new(),
                )
                .await;
            let run_id = next_request(&rx).await.run_id;
            let call_sequence = std::sync::atomic::AtomicUsize::new(0);
            let call = |name: &'static str, args| {
                let session = Arc::clone(&session);
                let turn = Arc::clone(&turn);
                let run_id = run_id.clone();
                let call_id = format!(
                    "external-{name}-{}",
                    call_sequence.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                );
                async move {
                    session
                        .external_tool_call(
                            &turn.sub_id,
                            &run_id,
                            call_id,
                            turn.config.multi_agent_v2.tool_namespace.clone(),
                            name.into(),
                            args,
                        )
                        .await
                        .expect("native child handler")
                }
            };
            let spawned = output_json(call("spawn_agent", json!({
            "task_name":"worker", "message":"inspect the native context", "executor":"antigravity"
        })).await);
            assert_eq!(spawned["task_name"], "/root/worker");
            let child_id = session
                .services
                .agent_control
                .resolve(
                    session.thread_id,
                    turn.parent_thread_id,
                    &turn.session_source,
                    "worker",
                )
                .await
                .expect("native child resolves");
            let child = manager
                .get_thread(child_id)
                .await
                .expect("native child exists");
            assert_eq!(
                child.config_snapshot().await.model_executor,
                ModelExecutor::Antigravity
            );
            let native_spawn = output_json(
                call(
                    "spawn_agent",
                    json!({
                        "task_name":"native_worker", "message":"use the native executor",
                        "executor":"codex", "model_provider":"openai", "model":"gpt-6-luna"
                    }),
                )
                .await,
            );
            assert_eq!(native_spawn["task_name"], "/root/native_worker");
            let native_id = session
                .services
                .agent_control
                .resolve(
                    session.thread_id,
                    turn.parent_thread_id,
                    &turn.session_source,
                    "native_worker",
                )
                .await
                .expect("native child resolves");
            let native_config = manager
                .get_thread(native_id)
                .await
                .expect("native child exists")
                .config_snapshot()
                .await;
            assert_eq!(native_config.model_executor, ModelExecutor::Codex);
            assert_eq!(native_config.model_provider_id, "openai");
            assert_eq!(native_config.model, "gpt-6-luna");
            assert_eq!(
                output_text(
                    call(
                        "send_message",
                        json!({"target":"worker", "message":"additional context"}),
                    )
                    .await
                ),
                ""
            );
            let listed = output_json(call("list_agents", json!({})).await);
            assert!(
                listed["agents"]
                    .as_array()
                    .expect("agents")
                    .iter()
                    .any(|agent| agent["agent_name"] == "/root/worker")
            );
            let interrupted =
                output_json(call("interrupt_agent", json!({"target":"worker"})).await);
            assert!(interrupted.get("previous_status").is_some());
            assert_eq!(
                output_text(
                    call(
                        "followup_task",
                        json!({"target":"worker", "message":"resume this task"}),
                    )
                    .await
                ),
                ""
            );
            let delivered: Vec<_> = manager
                .captured_ops()
                .into_iter()
                .filter_map(|(id, op)| match op {
                    Op::InterAgentCommunication { communication, .. } if id == child_id => {
                        Some(communication)
                    }
                    _ => None,
                })
                .collect();
            let expected: Vec<_> = [
                ("NEW_TASK", "inspect the native context", true),
                ("MESSAGE", "additional context", false),
                ("NEW_TASK", "resume this task", true),
            ]
            .into_iter()
            .map(|(message_type, message, trigger_turn)| {
                InterAgentCommunication::new(
                    AgentPath::root(),
                    AgentPath::root().join("worker").expect("child path"),
                    Vec::new(),
                    format!("Message Type: {message_type}\nTask name: /root/worker\nSender: /root\nPayload:\n{message}"),
                    trigger_turn,
                )
            })
            .collect();
            assert_eq!(delivered, expected, "native attributed plaintext delivery");
            session
                .input_queue
                .enqueue_mailbox_communication(
                    InterAgentCommunication::new(
                        AgentPath::try_from("/root/worker").expect("child path"),
                        AgentPath::root(),
                        Vec::new(),
                        "worker progress".into(),
                        false,
                    ),
                    Default::default(),
                )
                .await;
            let waited = tokio::time::timeout(
                Duration::from_secs(2),
                call("wait_agent", json!({"timeout_ms":10_000})),
            )
            .await
            .expect("native mailbox wakes wait");
            assert_eq!(output_json(waited)["timed_out"], false);
            session.abort_all_tasks(TurnAbortReason::Interrupted).await;
        });
    }

    #[test]
    fn native_relative_child_tools_keep_two_external_parent_trees_separate() {
        run_test(|| async {
            use codex_protocol::AgentPath;
            use codex_protocol::protocol::{InterAgentCommunication, Op};

            async fn parent(
                manager: &crate::ThreadManager,
            ) -> (Arc<Session>, Arc<TurnContext>, String) {
                let (mut session, turn, rx) =
                    crate::session::tests::make_session_and_context_with_auth_and_config_and_rx(
                        codex_login::CodexAuth::from_api_key("test"),
                        Vec::new(),
                        |config| {
                            config.model_executor = ModelExecutor::Antigravity;
                            config.model = Some("external-test-model".into());
                            config.multi_agent_v2.tool_namespace = Some("native_agents".into());
                            config
                                .features
                                .enable(codex_features::Feature::MultiAgentV2)
                                .expect("enable v2");
                        },
                    )
                    .await;
                let root = manager
                    .start_thread(crate::thread_manager::StartThreadOptions::new(
                        (*turn.config).clone(),
                    ))
                    .await
                    .expect("native root");
                let owned = Arc::get_mut(&mut session).expect("exclusive session");
                let control = manager.agent_control();
                owned.services.local_agent_runtime = control.runtime.clone();
                owned.services.agent_control = Arc::new(control);
                owned.thread_id = root.thread_id;
                session
                    .spawn_task(
                        Arc::clone(&turn),
                        Vec::new(),
                        crate::tasks::RegularTask::new(),
                    )
                    .await;
                let request = next_request(&rx).await;
                assert_eq!(request.thread_id, session.thread_id.to_string());
                (session, turn, request.run_id)
            }

            async fn call(
                session: &Session,
                turn: &TurnContext,
                run_id: &str,
                name: &str,
                arguments: serde_json::Value,
            ) -> ResponseItem {
                // Deliberately reuse each call ID in both parents, as separate CLI clients may do.
                let call_id = format!("parent-call-{name}");
                let output = session
                    .external_tool_call(
                        &turn.sub_id,
                        &run_id,
                        call_id.clone(),
                        turn.config.multi_agent_v2.tool_namespace.clone(),
                        name.into(),
                        arguments,
                    )
                    .await
                    .expect("native handler");
                assert!(matches!(&output, ResponseItem::FunctionCallOutput {
                    call_id: Some(actual), ..
                } if actual == &call_id));
                output
            }

            async fn child_event(child: &CodexThread, request: bool) {
                tokio::time::timeout(Duration::from_secs(10), async {
                    loop {
                        let event = child.next_event().await.expect("native child event");
                        if matches!(event.msg, EventMsg::ExternalExecutorRequest(_)) && request
                            || matches!(event.msg, EventMsg::TurnAborted(_)) && !request
                        {
                            break;
                        }
                    }
                })
                .await
                .expect("native child lifecycle boundary");
            }

            async fn history(session: &Session) -> Vec<ResponseItem> {
                session.clone_history().await.raw_items().cloned().collect()
            }

            async fn active_state(session: &Session) -> (String, bool, bool) {
                let active = session.active_turn.lock().await;
                let task = active
                    .as_ref()
                    .expect("active parent turn")
                    .task
                    .as_ref()
                    .expect("active external task");
                (
                    task.turn_context.sub_id.clone(),
                    task.cancellation_token.is_cancelled(),
                    session.is_interrupted(),
                )
            }

            let manager = crate::ThreadManager::with_models_provider_for_tests(
                codex_login::CodexAuth::from_api_key("test"),
                codex_model_provider_info::built_in_model_providers(None)["openai"].clone(),
            );
            let parents = [parent(&manager).await, parent(&manager).await];
            assert_ne!(parents[0].0.thread_id, parents[1].0.thread_id);
            let mut children = Vec::new();
            for (index, (session, turn, run_id)) in parents.iter().enumerate() {
                let spawned = output_json(
                    call(
                        session,
                        turn,
                        run_id,
                        "spawn_agent",
                        json!({
                            "task_name":"worker", "message":format!("task for parent {index}"),
                            "executor":"antigravity"
                        }),
                    )
                    .await,
                );
                assert_eq!(spawned["task_name"], "/root/worker");
                let id = session
                    .services
                    .agent_control
                    .resolve(
                        session.thread_id,
                        turn.parent_thread_id,
                        &turn.session_source,
                        "worker",
                    )
                    .await
                    .expect("parent-relative child");
                let child = manager.get_thread(id).await.expect("loaded native child");
                assert_eq!(
                    child.config_snapshot().await.parent_thread_id,
                    Some(session.thread_id)
                );
                child_event(&child, true).await;
                children.push((id, child));
            }
            assert_ne!(children[0].0, children[1].0);
            children[1].1.session.mark_interrupted();
            assert_ne!(
                children[0].1.agent_status().await,
                children[1].1.agent_status().await
            );

            for name in ["send_message", "list_agents", "interrupt_agent"] {
                for (index, (session, turn, run_id)) in parents.iter().enumerate() {
                    let other = 1 - index;
                    let parent_before = history(&parents[other].0).await;
                    let child_before = history(&children[other].1.session).await;
                    let other_parent_state = active_state(&parents[other].0).await;
                    let other_child_status = children[other].1.agent_status().await;
                    let own_child_status = children[index].1.agent_status().await;
                    let ops_before = manager.captured_ops().len();
                    let message = format!("context from parent {index}");
                    let args = match name {
                        "send_message" => json!({"target":"worker", "message":message}),
                        "list_agents" => json!({"path_prefix":"worker"}),
                        "interrupt_agent" => json!({"target":"worker"}),
                        _ => unreachable!(),
                    };
                    let output = call(session, turn, run_id, name, args).await;
                    let operations = manager.captured_ops();
                    let dispatched = &operations[ops_before..];
                    match name {
                        "send_message" => {
                            assert_eq!(output_text(output), "");
                            assert_eq!(dispatched.len(), 1);
                            assert_eq!(dispatched[0].0, children[index].0);
                            let Op::InterAgentCommunication { communication, .. } =
                                &dispatched[0].1
                            else {
                                panic!("attributed native communication expected");
                            };
                            assert_eq!(
                                communication,
                                &InterAgentCommunication::new(
                                    AgentPath::root(),
                                    AgentPath::root().join("worker").expect("child path"),
                                    Vec::new(),
                                    format!(
                                        "Message Type: MESSAGE\nTask name: /root/worker\nSender: /root\nPayload:\n{message}"
                                    ),
                                    false,
                                )
                            );
                        }
                        "list_agents" => {
                            assert_eq!(
                                output_json(output),
                                json!({"agents":[{
                                    "agent_name":"/root/worker", "agent_status":own_child_status
                                }]})
                            );
                            assert!(dispatched.is_empty());
                        }
                        "interrupt_agent" => {
                            assert_eq!(
                                output_json(output),
                                json!({"previous_status":own_child_status})
                            );
                            assert_eq!(dispatched.len(), 1);
                            assert_eq!(dispatched[0].0, children[index].0);
                            assert!(matches!(dispatched[0].1, Op::Interrupt));
                            child_event(&children[index].1, false).await;
                            assert!(children[index].1.session.is_interrupted());
                        }
                        _ => unreachable!(),
                    }
                    assert_eq!(history(&parents[other].0).await, parent_before);
                    assert_eq!(history(&children[other].1.session).await, child_before);
                    assert_eq!(active_state(&parents[other].0).await, other_parent_state);
                    assert_eq!(children[other].1.agent_status().await, other_child_status);
                    assert_eq!(
                        session
                            .services
                            .agent_control
                            .resolve(
                                session.thread_id,
                                turn.parent_thread_id,
                                &turn.session_source,
                                "worker",
                            )
                            .await
                            .expect("same parent-relative child"),
                        children[index].0
                    );
                }
            }
            for (session, _, _) in parents {
                session.abort_all_tasks(TurnAbortReason::Interrupted).await;
            }
        });
    }
}
