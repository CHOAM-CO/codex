use super::*;
use codex_extension_api::ExtensionData;
use codex_extension_api::TurnItemContributor;
use codex_protocol::ResponseItemId;
use codex_protocol::items::AgentMessageContent;
use pretty_assertions::assert_eq;
use std::sync::Arc;
use tracing_subscriber::prelude::*;

#[tokio::test]
async fn previous_runtime_restores_provider_and_usable_window_without_target_overrides() {
    let (session, mut destination) = crate::session::tests::make_session_and_context().await;
    let source_provider = codex_model_provider_info::create_oss_provider_with_base_url(
        "https://source.example/v1",
        codex_model_provider_info::WireApi::Responses,
    );
    let mut config = (*destination.config).clone();
    config
        .model_providers
        .insert("source".to_string(), source_provider.clone());
    config.model_context_window = Some(200_000);
    config.model_auto_compact_token_limit = Some(150_000);
    config.model_catalog = Some(codex_protocol::openai_models::ModelsResponse {
        models: vec![
            session
                .services
                .models_manager
                .get_model_info("source-model", &config.to_models_manager_config())
                .await,
        ],
    });
    destination.config = Arc::new(config);
    let runtime = codex_protocol::protocol::TurnModelRuntime {
        model_provider: "source".to_string(),
        model_executor: codex_protocol::external_executor::ModelExecutor::Codex,
        model_context_window: Some(500_000),
    };
    let previous = destination
        .with_model_runtime(
            "source-model".to_string(),
            &runtime,
            &session.services.models_manager,
        )
        .await
        .expect("restore previous runtime");
    assert_eq!(previous.model_runtime(), runtime);
    assert_eq!(previous.model_info().slug, "source-model");
    assert_eq!(previous.provider.info(), &source_provider);
    assert_eq!(previous.model_context_window(), Some(500_000));
    assert_eq!(previous.config.model_auto_compact_token_limit, None);
    assert_eq!(destination.config.model_context_window, Some(200_000));
    assert_ne!(previous.provider.info(), destination.provider.info());
}

async fn transition_test_session(
    source_url: String,
    destination_url: String,
) -> (Arc<Session>, Arc<TurnContext>, ResponseItem) {
    transition_test_session_with_capabilities(source_url, destination_url, false).await
}

async fn transition_test_session_with_capabilities(
    source_url: String,
    destination_url: String,
    destination_encrypted_content: bool,
) -> (Arc<Session>, Arc<TurnContext>, ResponseItem) {
    let mut source = codex_model_provider_info::built_in_model_providers(None)["openai"].clone();
    source.base_url = Some(format!("{source_url}/v1"));
    source.supports_websockets = false;
    source.stream_max_retries = Some(0);
    let mut destination = codex_model_provider_info::create_oss_provider_with_base_url(
        &format!("{destination_url}/v1"),
        codex_model_provider_info::WireApi::Responses,
    );
    destination.capabilities = Some(codex_model_provider_info::ModelProviderCapabilities {
        encrypted_content: Some(destination_encrypted_content),
        ..Default::default()
    });
    let (session, turn, _events) =
        crate::session::tests::make_session_and_context_with_auth_and_config_and_rx(
            CodexAuth::create_dummy_chatgpt_auth_for_testing(),
            Vec::new(),
            move |config| {
                config.model = Some("destination-model".to_string());
                config.model_provider_id = "destination".to_string();
                config.model_provider = destination.clone();
                config
                    .model_providers
                    .insert("destination".to_string(), destination);
                config.model_providers.insert("source".to_string(), source);
                config.chatgpt_base_url = source_url;
                config.model_context_window = Some(200_000);
                config.model_auto_compact_token_limit = Some(150_000);
                let _ = config.features.disable(Feature::TokenBudget);
            },
        )
        .await;
    let original =
        core_test_support::responses::user_message_item("Keep the original requirement.");
    session.replace_history(vec![original.clone()], None).await;
    session
        .state
        .lock()
        .await
        .set_token_info(Some(codex_protocol::protocol::TokenUsageInfo {
            total_token_usage: Default::default(),
            last_token_usage: codex_protocol::protocol::TokenUsage {
                input_tokens: 300_000,
                total_tokens: 300_000,
                ..Default::default()
            },
            model_context_window: Some(500_000),
        }));
    session
        .set_previous_turn_settings(Some(PreviousTurnSettings {
            model: "source-model".to_string(),
            model_runtime: Some(codex_protocol::protocol::TurnModelRuntime {
                model_provider: "source".to_string(),
                model_executor: codex_protocol::external_executor::ModelExecutor::Codex,
                model_context_window: Some(500_000),
            }),
            cyber_access_program: None,
            comp_hash: None,
            realtime_active: Some(false),
        }))
        .await;
    (session, turn, original)
}

#[test_case::test_case(false; "destination does not support encryption")]
#[test_case::test_case(true; "destination uses its own encryption")]
#[tokio::test]
async fn provider_downshift_compacts_once_with_source_before_destination_sampling(
    destination_encrypted_content: bool,
) {
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    let source_requests = responses::mount_sse_once(
        &source,
        responses::sse(vec![
            responses::ev_assistant_message("summary", "Preserved the original requirement."),
            responses::ev_completed("compacted"),
        ]),
    )
    .await;
    let (session, turn, _) = transition_test_session_with_capabilities(
        source.uri(),
        destination.uri(),
        destination_encrypted_content,
    )
    .await;
    let mut client = session.services.model_client.new_session();
    run_pre_sampling_compact(&session, &turn, &mut client, &CancellationToken::new())
        .await
        .expect("source compaction succeeds");
    let calls = source_requests.requests();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].body_json()["model"], "source-model");
    assert!(
        calls[0]
            .message_input_texts("user")
            .iter()
            .any(|text| text == "Keep the original requirement.")
    );
    assert!(
        responses::received_responses_requests(&destination)
            .await
            .is_empty()
    );
    let history = session.clone_history().await;
    assert!(
        history
            .raw_items()
            .all(|item| !matches!(item, ResponseItem::Compaction { .. }))
    );
    assert!(history.raw_items().any(|item| matches!(item, ResponseItem::Message { content, .. } if content.iter().any(|content| matches!(content, ContentItem::InputText { text } if text.contains("Preserved the original requirement."))))));
    assert_eq!(turn.config.model_provider_id, "destination");
    assert!(session.get_total_token_usage().await < 150_000);
}

#[tokio::test]
async fn failed_provider_downshift_preserves_history_and_never_calls_destination() {
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    let source_requests = responses::mount_response_once(
        &source,
        wiremock::ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": {"message": "Source context does not fit", "code": "context_length_exceeded"}
        })),
    )
    .await;
    let (session, turn, original) = transition_test_session(source.uri(), destination.uri()).await;
    let mut client = session.services.model_client.new_session();
    let result =
        run_pre_sampling_compact(&session, &turn, &mut client, &CancellationToken::new()).await;
    assert!(result.is_err());
    let calls = responses::received_responses_requests(&source).await;
    assert_eq!(source_requests.requests().len(), 1);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].body_json()["model"], "source-model");
    assert!(
        responses::received_responses_requests(&destination)
            .await
            .is_empty()
    );
    assert_eq!(
        session
            .clone_history()
            .await
            .raw_items()
            .cloned()
            .collect::<Vec<_>>(),
        vec![original]
    );
    assert_eq!(
        session.previous_turn_settings().await.unwrap().model,
        "source-model"
    );
}

#[tokio::test]
async fn partial_downshift_summary_is_not_committed_when_stream_fails() {
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    responses::mount_sse_once(
        &source,
        responses::sse(vec![responses::ev_assistant_message(
            "partial",
            "Incomplete summary must not enter history.",
        )]),
    )
    .await;
    let (session, turn, original) = transition_test_session(source.uri(), destination.uri()).await;
    let mut client = session.services.model_client.new_session();
    assert!(
        run_pre_sampling_compact(&session, &turn, &mut client, &CancellationToken::new())
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
        vec![original]
    );
    assert!(
        responses::received_responses_requests(&destination)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn empty_downshift_summary_preserves_source_context_size() {
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    responses::mount_sse_once(
        &source,
        responses::sse(vec![responses::ev_completed("empty-summary")]),
    )
    .await;
    let (session, turn, original) = transition_test_session(source.uri(), destination.uri()).await;
    let context_tokens = session.get_total_token_usage().await;
    let mut client = session.services.model_client.new_session();
    assert!(
        run_pre_sampling_compact(&session, &turn, &mut client, &CancellationToken::new())
            .await
            .is_err()
    );
    assert_eq!(session.get_total_token_usage().await, context_tokens);
    assert_eq!(
        session
            .clone_history()
            .await
            .raw_items()
            .cloned()
            .collect::<Vec<_>>(),
        vec![original]
    );
    assert!(
        responses::received_responses_requests(&destination)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn same_model_slug_uses_previous_provider_catalog_for_compaction() {
    use core_test_support::responses;
    // Unlike the general fixture, this server must not install a default empty catalog.
    let source = wiremock::MockServer::start().await;
    let destination = responses::start_mock_server().await;
    let (session, mut turn, _) = transition_test_session(source.uri(), destination.uri()).await;
    let mut config = (*turn.config).clone();
    let source_provider = config.model_providers.get_mut("source").unwrap();
    source_provider.model_catalog_url = Some(format!("{}/v1/models", source.uri()).into());
    source_provider.experimental_bearer_token = Some("test-catalog-token".into());
    config
        .features
        .enable(Feature::ApiKeyModelDiscovery)
        .unwrap();
    Arc::get_mut(&mut turn).unwrap().config = Arc::new(config);
    let mut source_model = turn.model_info().as_ref().clone();
    source_model.supports_reasoning_summary_parameter = false;
    source_model.context_window = Some(500_000);
    source_model.max_context_window = Some(500_000);
    let catalog_requests = responses::mount_models_once(
        &source,
        codex_protocol::openai_models::ModelsResponse {
            models: vec![source_model],
        },
    )
    .await;
    let mut previous = session.previous_turn_settings().await.unwrap();
    previous.model = turn.model_info().slug.clone();
    let restored = turn
        .with_model_runtime(
            previous.model.clone(),
            previous.model_runtime.as_ref().unwrap(),
            &session.services.models_manager,
        )
        .await
        .unwrap();
    assert!(!catalog_requests.requests().is_empty());
    assert!(!restored.model_info().supports_reasoning_summary_parameter);
    assert!(turn.model_info().supports_reasoning_summary_parameter);
    assert_eq!(restored.model_info().slug, turn.model_info().slug);
    assert_eq!(
        restored.provider.info().base_url.as_deref(),
        Some(format!("{}/v1", source.uri()).as_str())
    );
    assert_eq!(restored.model_context_window(), Some(500_000));
    assert!(
        responses::received_responses_requests(&source)
            .await
            .is_empty()
    );
    assert!(
        responses::received_responses_requests(&destination)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn unsupported_previous_executor_preserves_history_without_http_sampling() {
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    let (session, turn, original) = transition_test_session(source.uri(), destination.uri()).await;
    let mut previous = session.previous_turn_settings().await.unwrap();
    previous.model_runtime.as_mut().unwrap().model_executor =
        codex_protocol::external_executor::ModelExecutor::Antigravity;
    session
        .set_previous_turn_settings(Some(previous.clone()))
        .await;
    let mut client = session.services.model_client.new_session();
    let error = run_pre_sampling_compact(&session, &turn, &mut client, &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(
        error.details(),
        CodexErrorDetails::UnsupportedOperation(_)
    ));
    assert_eq!(
        session
            .clone_history()
            .await
            .raw_items()
            .cloned()
            .collect::<Vec<_>>(),
        vec![original]
    );
    assert_eq!(session.previous_turn_settings().await, Some(previous));
    assert!(
        responses::received_responses_requests(&source)
            .await
            .is_empty()
    );
    assert!(
        responses::received_responses_requests(&destination)
            .await
            .is_empty()
    );
    assert_eq!(turn.config.model_provider_id, "destination");
}

#[tokio::test]
async fn manual_compaction_rejects_external_executor_without_http_sampling() {
    use crate::tasks::SessionTask;
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    let (session, mut turn, original) =
        transition_test_session(source.uri(), destination.uri()).await;
    Arc::make_mut(&mut Arc::get_mut(&mut turn).unwrap().config).model_executor =
        codex_protocol::external_executor::ModelExecutor::Antigravity;
    let error = Arc::new(crate::tasks::CompactTask)
        .run(session.clone(), turn, Vec::new(), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(
        error.details(),
        CodexErrorDetails::UnsupportedOperation(_)
    ));
    assert_eq!(
        session
            .clone_history()
            .await
            .raw_items()
            .cloned()
            .collect::<Vec<_>>(),
        vec![original]
    );
    assert!(
        responses::received_responses_requests(&source)
            .await
            .is_empty()
    );
    assert!(
        responses::received_responses_requests(&destination)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn oversized_downshift_summary_stops_after_one_source_request() {
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    let source_requests = responses::mount_sse_once(
        &source,
        responses::sse(vec![
            responses::ev_assistant_message("summary", &"summary detail ".repeat(60_000)),
            responses::ev_completed("compacted"),
        ]),
    )
    .await;
    let (session, turn, _) = transition_test_session(source.uri(), destination.uri()).await;
    let mut client = session.services.model_client.new_session();
    let error = run_pre_sampling_compact(&session, &turn, &mut client, &CancellationToken::new())
        .await
        .expect_err("an oversized summary must stop the transition");
    assert!(matches!(
        error.details(),
        CodexErrorDetails::InvalidRequest(_)
    ));
    assert_eq!(source_requests.requests().len(), 1);
    assert_eq!(
        responses::received_responses_requests(&source).await.len(),
        1
    );
    assert!(
        responses::received_responses_requests(&destination)
            .await
            .is_empty()
    );
    assert!(session.get_total_token_usage().await >= 150_000);
}

struct RewriteAgentMessageContributor;

impl TurnItemContributor for RewriteAgentMessageContributor {
    fn contribute<'a>(
        &'a self,
        _thread_store: &'a ExtensionData,
        _turn_store: &'a ExtensionData,
        item: &'a mut TurnItem,
    ) -> codex_extension_api::ExtensionFuture<'a, Result<(), String>> {
        Box::pin(async move {
            if let TurnItem::AgentMessage(agent_message) = item {
                agent_message.content = vec![AgentMessageContent::Text {
                    text: "plan contributed assistant text".to_string(),
                }];
            }
            Ok(())
        })
    }
}

fn assistant_output_text(text: &str) -> ResponseItem {
    ResponseItem::Message {
        status: None,
        encrypted_content: None,
        id: Some(ResponseItemId::with_suffix("msg", "1")),
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            annotations: None,
            logprobs: None,
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

#[test]
fn post_sampling_token_estimate_is_disabled_by_always_on_sinks() {
    let feedback = codex_feedback::CodexFeedback::new();
    let subscriber = tracing_subscriber::registry()
        .with(feedback.logger_layer())
        .with(tracing_subscriber::fmt::layer().with_filter(codex_state::log_db::default_filter()));

    static METADATA: tracing::Metadata<'static> = tracing::metadata! {
        name: "post sampling token estimate filter probe",
        target: POST_SAMPLING_TOKEN_ESTIMATE_TARGET,
        level: tracing::Level::TRACE,
        fields: &["turn_id", "estimated_token_count", "message"],
        callsite: &CALLSITE,
        kind: tracing::metadata::Kind::EVENT.hint(),
    };
    static CALLSITE: tracing::callsite::DefaultCallsite =
        tracing::callsite::DefaultCallsite::new(&METADATA);

    assert!(tracing::Subscriber::register_callsite(&subscriber, &METADATA).is_never());
}

#[tokio::test]
async fn plan_mode_uses_contributed_turn_item_for_last_agent_message() {
    let (mut session, turn_context) = crate::session::tests::make_session_and_context().await;
    let mut builder = codex_extension_api::ExtensionRegistryBuilder::new();
    builder.turn_item_contributor(Arc::new(RewriteAgentMessageContributor));
    session.services.extensions = Arc::new(builder.build());
    let turn_store = ExtensionData::new(turn_context.sub_id.clone());
    let mut state = PlanModeStreamState::new(&turn_context.sub_id);
    let mut last_agent_message = None;
    let item = assistant_output_text("original assistant text");

    let step_context = StepContext::for_test(Arc::new(turn_context));
    let handled = handle_assistant_item_done_in_plan_mode(
        &session,
        &step_context,
        &turn_store,
        &item,
        &mut state,
        /*previously_active_item*/ None,
        &mut last_agent_message,
    )
    .await;

    assert!(handled);
    assert_eq!(
        last_agent_message.as_deref(),
        Some("plan contributed assistant text")
    );
}

#[test]
fn realtime_user_verification_notice_excludes_request_payload() {
    let event = EventMsg::ElicitationRequest(codex_protocol::approvals::ElicitationRequestEvent {
        turn_id: None,
        server_name: "private-server-name".to_string(),
        id: codex_protocol::mcp::RequestId::String("private-request-id".to_string()),
        request: codex_protocol::approvals::ElicitationRequest::UserVerification {
            meta: None,
            title: "private-title".to_string(),
            description: "private-description".to_string(),
            challenge: "private-challenge".to_string(),
        },
    });
    assert_eq!(
        realtime_text_for_event(&event),
        Some(RealtimeEventText::Handoff(
            "<user_verification_notice>User verification is required. Please respond in the app.</user_verification_notice>".to_string(),
            None,
        )),
    );
}
