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
    transition_test_session_with_input_counting(
        source_url,
        destination_url,
        destination_encrypted_content,
        false,
    )
    .await
}

async fn transition_test_session_with_input_counting(
    source_url: String,
    destination_url: String,
    destination_encrypted_content: bool,
    count_input: bool,
) -> (Arc<Session>, Arc<TurnContext>, ResponseItem) {
    transition_test_session_with_count_scope(
        source_url,
        destination_url,
        destination_encrypted_content,
        count_input,
        codex_protocol::config_types::AutoCompactTokenLimitScope::Total,
    )
    .await
}

async fn transition_test_session_with_count_scope(
    source_url: String,
    destination_url: String,
    destination_encrypted_content: bool,
    count_input: bool,
    scope: codex_protocol::config_types::AutoCompactTokenLimitScope,
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
    destination.input_tokens_endpoint = count_input.then(|| "responses/input_tokens".to_owned());
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
                config.model_auto_compact_token_limit_scope = scope;
                config.model_post_turn_compact_threshold_percent = 0;
                config.update_plan_enabled = true;
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

async fn use_below_destination_budget(session: &Session) {
    session
        .state
        .lock()
        .await
        .set_token_info(Some(codex_protocol::protocol::TokenUsageInfo {
            total_token_usage: Default::default(),
            last_token_usage: codex_protocol::protocol::TokenUsage {
                input_tokens: 100_000,
                total_tokens: 100_000,
                ..Default::default()
            },
            model_context_window: Some(500_000),
        }));
}

fn transition_input(text: String) -> Vec<TurnInput> {
    vec![TurnInput::UserInput {
        content: vec![UserInput::Text {
            text,
            text_elements: Vec::new(),
        }],
        client_id: None,
        metadata: Default::default(),
    }]
}

async fn mount_input_counts(server: &wiremock::MockServer, counts: Vec<i64>) {
    let count_len = counts.len() as u64;
    let next = std::sync::atomic::AtomicUsize::new(0);
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/responses/input_tokens"))
        .respond_with(move |_: &wiremock::Request| {
            let index = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let Some(count) = counts.get(index) else {
                return wiremock::ResponseTemplate::new(400);
            };
            wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "response.input_tokens", "input_tokens": count,
            }))
        })
        .expect(count_len)
        .mount(server)
        .await;
}

#[tokio::test]
async fn counted_pre_sampling_preserves_ready_candidate_from_previous_model_until_counting() {
    use crate::context::world_state::WorldState;
    use codex_protocol::protocol::TokenUsage;
    use core_test_support::responses;
    let server = responses::start_mock_server().await;
    let mut provider = codex_model_provider_info::create_oss_provider_with_base_url(
        &format!("{}/v1", server.uri()),
        codex_model_provider_info::WireApi::Responses,
    );
    provider.input_tokens_endpoint = Some("responses/input_tokens".to_owned());
    provider.capabilities = Some(codex_model_provider_info::ModelProviderCapabilities {
        encrypted_content: Some(true),
        remote_compaction: Some(codex_model_provider_info::RemoteCompactionSupport::V2),
        ..Default::default()
    });
    let (session, source_turn, _events) =
        crate::session::tests::make_session_and_context_with_auth_and_config_and_rx(
            CodexAuth::create_dummy_chatgpt_auth_for_testing(),
            Vec::new(),
            move |config| {
                config.model = Some("source-model".to_owned());
                config.model_provider_id = "counted-provider".to_owned();
                config.model_provider = provider.clone();
                config
                    .model_providers
                    .insert("counted-provider".to_owned(), provider);
                config.model_context_window = Some(500_000);
                config.model_auto_compact_token_limit = Some(450_000);
                config.async_compaction_threshold_percent = 50;
                // The fallback model catalog caps unknown models at 272k. Give this
                // source a real 500k catalog window so 300k can launch speculation.
                let mut source_model =
                    codex_models_manager::test_support::construct_model_info_offline_for_tests(
                        "source-model",
                        &config.to_models_manager_config(),
                    );
                source_model.context_window = Some(500_000);
                source_model.max_context_window = Some(500_000);
                config.model_catalog = Some(codex_protocol::openai_models::ModelsResponse {
                    models: vec![source_model],
                });
                config.features.enable(Feature::AsyncCompaction).unwrap();
                config.features.disable(Feature::TokenBudget).unwrap();
            },
        )
        .await;
    let original = responses::user_message_item("Preserve this source context.");
    session.replace_history(vec![original.clone()], None).await;
    session
        .update_token_usage_info(
            &source_turn,
            Some(&TokenUsage {
                input_tokens: 300_000,
                total_tokens: 300_000,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    session
        .set_previous_turn_settings(Some(PreviousTurnSettings {
            model: "source-model".to_owned(),
            model_runtime: Some(source_turn.model_runtime()),
            cyber_access_program: None,
            comp_hash: None,
            realtime_active: Some(false),
        }))
        .await;
    let prepared = responses::mount_sse_once(
        &server,
        responses::sse(vec![
            serde_json::json!({"type":"response.output_item.done", "item":{
                "type":"compaction", "encrypted_content":"source-prepared-summary"
            }}),
            responses::ev_completed("source-prepared"),
        ]),
    )
    .await;
    let cancellation = CancellationToken::new();
    let source_step = session
        .capture_step_context(source_turn.clone(), &cancellation)
        .await
        .unwrap();
    let source_status =
        super::super::context_window::context_window_token_status(&session, &source_turn).await;
    assert!(super::super::async_compaction::enabled(&source_turn));
    assert_eq!(
        source_step.settings.model_info.resolved_context_window(),
        Some(500_000)
    );
    assert!(!source_status.token_limit_reached);
    assert!(source_status.active_context_tokens >= 250_000);
    super::super::async_compaction::maybe_start(
        &session,
        &source_step,
        &Arc::new(WorldState::default()),
        CompactionPhase::MidTurn,
        &cancellation,
        &source_status,
        (
            Some("source-routing".to_owned()),
            session.services.model_client.auth_owner_generation(),
        ),
    )
    .await;
    super::super::async_compaction::wait_for_background_compaction_for_test(&session).await;
    assert_eq!(prepared.requests().len(), 1);
    assert_eq!(prepared.requests()[0].body_json()["model"], "source-model");

    let mut target_runtime = source_turn.model_runtime();
    target_runtime.model_context_window = Some(200_000);
    let target_turn = Arc::new(
        source_turn
            .with_model_runtime(
                "destination-model".to_owned(),
                &target_runtime,
                &session.services.models_manager,
            )
            .await
            .unwrap(),
    );
    let target_status =
        super::super::context_window::context_window_token_status(&session, &target_turn).await;
    assert!(target_status.token_limit_reached);
    let history_before = session.clone_history().await;
    let usage_before = session.state.lock().await.token_info();
    let mut client = session.services.model_client.new_session();
    assert!(
        !run_pre_sampling_compact(&session, &target_turn, &mut client, &cancellation,)
            .await
            .unwrap()
    );
    assert_eq!(prepared.requests().len(), 1);
    assert_eq!(
        session
            .clone_history()
            .await
            .raw_items()
            .cloned()
            .collect::<Vec<_>>(),
        history_before.raw_items().cloned().collect::<Vec<_>>()
    );
    assert_eq!(session.state.lock().await.token_info(), usage_before);

    mount_input_counts(&server, vec![500]).await;
    let target_step = session
        .capture_step_context(target_turn.clone(), &cancellation)
        .await
        .unwrap();
    let count = client
        .count_input_tokens(
            &Prompt {
                input: vec![original],
                ..Default::default()
            },
            &target_step.settings.model_info,
            &target_step.session_telemetry,
            None,
            target_turn.reasoning_summary(),
            None,
            &CodexResponsesMetadata::new(
                "test-installation".to_owned(),
                "test-session".to_owned(),
                session.thread_id.to_string(),
                "test-window".to_owned(),
            ),
        )
        .await
        .unwrap();
    assert_eq!(count, Some(500));
    assert!(
        super::super::async_compaction::take_ready(&session, &target_turn)
            .await
            .is_some()
    );
    let calls = server
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .filter(|request| request.method.as_str() == "POST")
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[1].url.path(), "/v1/responses/input_tokens");
    assert_eq!(
        calls[1].body_json::<serde_json::Value>().unwrap()["model"],
        "destination-model"
    );
    assert_eq!(prepared.requests().len(), 1);
    super::super::async_compaction::shutdown(&session).await;
}

#[test_case::test_case(false; "target count overrides larger source usage")]
#[test_case::test_case(true; "target count triggers source compaction and recount")]
#[tokio::test]
async fn counted_input_uses_target_budget_before_generation(compact: bool) {
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    let source_requests = responses::mount_sse_once(
        &source,
        responses::sse(vec![
            responses::ev_assistant_message(
                "summary",
                "Preserved source requirement in the summary.",
            ),
            responses::ev_completed("compacted"),
        ]),
    )
    .await;
    mount_input_counts(
        &destination,
        if compact {
            vec![200_000, 500]
        } else {
            vec![500]
        },
    )
    .await;
    let destination_requests = responses::mount_sse_once(
        &destination,
        responses::sse(vec![
            responses::ev_assistant_message("answer", "Finished after counting."),
            responses::ev_completed("answered"),
        ]),
    )
    .await;
    let (session, turn, _) =
        transition_test_session_with_input_counting(source.uri(), destination.uri(), false, true)
            .await;
    if compact {
        // The exact target count exceeds its budget even though inherited usage does not.
        use_below_destination_budget(&session).await;
    }
    let input = "Preserve this exact user input once.";
    let result = run_turn(
        session.clone(),
        turn,
        transition_input(input.to_owned()),
        &mut McpStartupRequirements::default(),
        None,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    assert_eq!(result.as_deref(), Some("Finished after counting."));
    assert_eq!(source_requests.requests().len(), usize::from(compact));
    let calls = destination.received_requests().await.unwrap();
    let counts = calls
        .iter()
        .filter(|request| request.url.path().ends_with("/input_tokens"))
        .collect::<Vec<_>>();
    assert_eq!(counts.len(), if compact { 2 } else { 1 });
    let count_bodies = counts
        .iter()
        .map(|request| request.body_json::<serde_json::Value>().unwrap())
        .collect::<Vec<_>>();
    for body in &count_bodies {
        assert_eq!(body["model"], "destination-model");
        assert_eq!(
            body["input"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|item| item["content"].as_array().into_iter().flatten())
                .filter(|part| part["text"] == input)
                .count(),
            1
        );
    }
    assert_eq!(destination_requests.requests().len(), 1);
    assert_eq!(
        count_bodies.last().unwrap()["input"],
        destination_requests.requests()[0].body_json()["input"]
    );
    if compact {
        assert_eq!(
            source_requests.requests()[0].body_json()["model"],
            "source-model"
        );
        assert!(
            count_bodies[1]["input"]
                .to_string()
                .contains("Preserved source requirement in the summary.")
        );
    }
}

#[test_case::test_case(401; "authentication failure")]
#[test_case::test_case(200; "invalid count response")]
#[tokio::test]
async fn counted_input_failure_preserves_usage_and_never_generates(status: u16) {
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/responses/input_tokens"))
        .respond_with(
            wiremock::ResponseTemplate::new(status)
                .set_body_json(serde_json::json!({"error":{"message":"count unavailable"}})),
        )
        .expect(1)
        .mount(&destination)
        .await;
    let (session, turn, original) =
        transition_test_session_with_input_counting(source.uri(), destination.uri(), false, true)
            .await;
    let before = session.state.lock().await.token_info();
    let previous = session.previous_turn_settings().await;
    let result = run_turn(
        session.clone(),
        turn,
        transition_input("Keep this request.".to_owned()),
        &mut McpStartupRequirements::default(),
        None,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    assert_eq!(result, None);
    assert_eq!(session.state.lock().await.token_info(), before);
    assert_eq!(session.previous_turn_settings().await, previous);
    assert!(
        session
            .clone_history()
            .await
            .raw_items()
            .any(|item| item == &original)
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

#[test_case::test_case(true; "count fails before sampling")]
#[test_case::test_case(false; "sampling fails before output")]
#[tokio::test]
async fn counted_provider_failure_keeps_source_prefix_and_usage(count_fails: bool) {
    use codex_protocol::config_types::AutoCompactTokenLimitScope;
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    if count_fails {
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/responses/input_tokens"))
            .respond_with(
                wiremock::ResponseTemplate::new(400)
                    .set_body_json(serde_json::json!({"error":{"message":"count unavailable"}})),
            )
            .expect(1)
            .mount(&destination)
            .await;
    } else {
        mount_input_counts(&destination, vec![500]).await;
        responses::mount_response_once(
            &destination,
            wiremock::ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "error": {"type":"invalid_request_error", "message":"target refused prompt"}
            })),
        )
        .await;
    }
    let (session, turn, _) = transition_test_session_with_count_scope(
        source.uri(),
        destination.uri(),
        false,
        true,
        AutoCompactTokenLimitScope::BodyAfterPrefix,
    )
    .await;
    let (usage_before, prefix_before, ids_before, number_before) = {
        let mut state = session.state.lock().await;
        let usage = state.token_info().unwrap().last_token_usage;
        state.ensure_auto_compact_window_server_prefill_from_usage(&usage);
        (
            state.token_info(),
            state.auto_compact_window_snapshot(),
            state.auto_compact_window_ids(),
            state.auto_compact_window_number(),
        )
    };
    assert_eq!(prefix_before.prefill_input_tokens, Some(300_000));
    let previous = session.previous_turn_settings().await;
    let result = run_turn(
        session.clone(),
        turn,
        transition_input("Retain the source state if this request fails.".to_owned()),
        &mut McpStartupRequirements::default(),
        None,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    assert_eq!(result, None);
    let state = session.state.lock().await;
    assert_eq!(state.token_info(), usage_before);
    assert_eq!(state.auto_compact_window_snapshot(), prefix_before);
    assert_eq!(state.auto_compact_window_ids(), ids_before);
    assert_eq!(state.auto_compact_window_number(), number_before);
    drop(state);
    assert_eq!(session.previous_turn_settings().await, previous);
    assert!(
        responses::received_responses_requests(&source)
            .await
            .is_empty()
    );
    assert_eq!(
        responses::received_responses_requests(&destination)
            .await
            .len(),
        usize::from(!count_fails)
    );
}

#[tokio::test]
async fn counted_provider_output_replaces_prefix_without_resetting_usage_or_window() {
    use codex_protocol::config_types::AutoCompactTokenLimitScope;
    use codex_protocol::protocol::TokenUsage;
    use codex_protocol::protocol::TokenUsageInfo;
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    mount_input_counts(&destination, vec![500]).await;
    let generation = responses::mount_sse_once(
        &destination,
        responses::sse(vec![
            responses::ev_assistant_message("answer", "The target answered."),
            serde_json::json!({
                "type":"response.completed",
                "response": {
                    "id":"target-answer",
                    "usage":{"input_tokens":500,"output_tokens":20,"total_tokens":520}
                }
            }),
        ]),
    )
    .await;
    let (session, turn, _) = transition_test_session_with_count_scope(
        source.uri(),
        destination.uri(),
        false,
        true,
        AutoCompactTokenLimitScope::BodyAfterPrefix,
    )
    .await;
    let source_usage = TokenUsage {
        input_tokens: 300_000,
        output_tokens: 100,
        total_tokens: 300_100,
        ..Default::default()
    };
    let (ids_before, number_before) = {
        let mut state = session.state.lock().await;
        state.set_token_info(Some(TokenUsageInfo {
            total_token_usage: source_usage.clone(),
            last_token_usage: source_usage.clone(),
            model_context_window: Some(500_000),
        }));
        state.ensure_auto_compact_window_server_prefill_from_usage(&source_usage);
        (
            state.auto_compact_window_ids(),
            state.auto_compact_window_number(),
        )
    };
    let result = run_turn(
        session.clone(),
        turn.clone(),
        transition_input("Continue with the target tokenizer.".to_owned()),
        &mut McpStartupRequirements::default(),
        None,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    assert_eq!(result.as_deref(), Some("The target answered."));
    let usage_after = {
        let state = session.state.lock().await;
        assert_eq!(
            state.auto_compact_window_snapshot().prefill_input_tokens,
            Some(500)
        );
        assert_eq!(state.auto_compact_window_ids(), ids_before);
        assert_eq!(state.auto_compact_window_number(), number_before);
        let info = state.token_info().unwrap();
        assert_eq!(
            info.last_token_usage,
            TokenUsage {
                input_tokens: 500,
                output_tokens: 20,
                total_tokens: 520,
                ..Default::default()
            }
        );
        assert_eq!(
            info.total_token_usage,
            TokenUsage {
                input_tokens: 300_500,
                output_tokens: 120,
                total_tokens: 300_620,
                ..Default::default()
            }
        );
        info
    };
    assert_eq!(
        session
            .previous_turn_settings()
            .await
            .unwrap()
            .model_runtime,
        Some(turn.model_runtime())
    );
    // A subsequent counted request on this runtime subtracts the target's observed
    // prefix, not the larger source prefix, without recording the count as usage.
    let status = super::super::context_window::context_window_token_status_with_input_tokens(
        &session,
        &turn,
        &turn.model_info(),
        750,
        true,
    )
    .await;
    assert_eq!(status.active_context_tokens, 750);
    assert_eq!(status.auto_compact_window_prefill_tokens, Some(500));
    assert_eq!(status.auto_compact_scope_tokens, 250);
    assert!(!status.token_limit_reached);
    assert_eq!(session.state.lock().await.token_info(), Some(usage_after));
    assert_eq!(generation.requests().len(), 1);
    assert!(
        responses::received_responses_requests(&source)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn counted_tool_completion_without_usage_recounts_before_considering_stale_source_usage() {
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    mount_input_counts(&destination, vec![500, 500]).await;
    let generation = responses::mount_sse_sequence(
        &destination,
        vec![
            responses::sse(vec![
                responses::ev_function_call(
                    "plan-once",
                    "update_plan",
                    r#"{"plan":[{"step":"Record the result","status":"completed"}]}"#,
                ),
                serde_json::json!({"type":"response.completed","response":{"id":"tool-request"}}),
            ]),
            responses::sse(vec![
                responses::ev_assistant_message("answer", "Continued without compaction."),
                serde_json::json!({"type":"response.completed","response":{"id":"followup"}}),
            ]),
        ],
    )
    .await;
    let (session, turn, _) =
        transition_test_session_with_input_counting(source.uri(), destination.uri(), false, true)
            .await;
    let usage_before = session.state.lock().await.token_info();
    assert_eq!(
        usage_before.as_ref().unwrap().last_token_usage.input_tokens,
        300_000
    );
    let result = run_turn(
        session.clone(),
        turn,
        transition_input("Record the plan, then continue.".to_owned()),
        &mut McpStartupRequirements::default(),
        None,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    assert_eq!(result.as_deref(), Some("Continued without compaction."));
    assert_eq!(session.state.lock().await.token_info(), usage_before);
    assert_eq!(generation.requests().len(), 2);
    assert!(
        responses::received_responses_requests(&source)
            .await
            .is_empty()
    );
    let requests = destination.received_requests().await.unwrap();
    assert_eq!(
        requests
            .iter()
            .map(|request| request.url.path())
            .collect::<Vec<_>>(),
        vec![
            "/v1/responses/input_tokens",
            "/v1/responses",
            "/v1/responses/input_tokens",
            "/v1/responses",
        ]
    );
    let followup_count = requests[2].body_json::<serde_json::Value>().unwrap();
    let tool_outputs = followup_count["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "function_call_output" && item["call_id"] == "plan-once")
        .collect::<Vec<_>>();
    assert_eq!(tool_outputs.len(), 1);
    assert_eq!(tool_outputs[0]["output"], "Plan updated");
    assert_eq!(
        followup_count["input"],
        generation.requests()[1].body_json()["input"]
    );
}

#[tokio::test]
async fn counted_input_still_over_budget_after_compaction_stops_without_generation() {
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    let source_requests = responses::mount_sse_once(
        &source,
        responses::sse(vec![
            responses::ev_assistant_message("summary", "Summary of previous work."),
            responses::ev_completed("compacted"),
        ]),
    )
    .await;
    mount_input_counts(&destination, vec![200_000, 200_000]).await;
    let (session, turn, _) =
        transition_test_session_with_input_counting(source.uri(), destination.uri(), false, true)
            .await;
    let input = "New input too large for the target. ".repeat(6_000);
    let result = run_turn(
        session.clone(),
        turn,
        transition_input(input.clone()),
        &mut McpStartupRequirements::default(),
        None,
        CancellationToken::new(),
    )
    .await;
    let error = result.unwrap_err();
    assert!(matches!(
        error.details(),
        CodexErrorDetails::InvalidRequest(_)
    ));
    assert_eq!(source_requests.requests().len(), 1);
    assert!(
        responses::received_responses_requests(&destination)
            .await
            .is_empty()
    );
    let history = session.clone_history().await;
    assert_eq!(history.raw_items().filter(|item| matches!(item, ResponseItem::Message { content, .. }
        if content.iter().any(|part| matches!(part, ContentItem::InputText {text} if text == &input)))).count(), 1);
}

#[tokio::test]
async fn counted_tool_result_compacts_current_model_and_recounts_before_followup() {
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    mount_input_counts(&destination, vec![500, 200_000, 500]).await;
    let generation = responses::mount_sse_sequence(
        &destination,
        vec![
            responses::sse(vec![
                responses::ev_function_call(
                    "plan-once",
                    "update_plan",
                    r#"{"plan":[{"step":"Record the result","status":"completed"}]}"#,
                ),
                responses::ev_completed("tool-request"),
            ]),
            responses::sse(vec![
                responses::ev_assistant_message("summary", "The plan was already recorded."),
                responses::ev_completed("current-compaction"),
            ]),
            responses::sse(vec![
                responses::ev_assistant_message("answer", "Continued after the tool result."),
                responses::ev_completed("followup"),
            ]),
        ],
    )
    .await;
    let (session, turn, _) =
        transition_test_session_with_input_counting(source.uri(), destination.uri(), false, true)
            .await;
    session
        .set_previous_turn_settings(Some(PreviousTurnSettings {
            model: "destination-model".to_owned(),
            model_runtime: Some(turn.model_runtime()),
            cyber_access_program: None,
            comp_hash: None,
            realtime_active: Some(false),
        }))
        .await;
    use_below_destination_budget(&session).await;
    let result = run_turn(
        session,
        turn,
        transition_input("Record the plan, then continue.".to_owned()),
        &mut McpStartupRequirements::default(),
        None,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    assert_eq!(result.as_deref(), Some("Continued after the tool result."));
    assert!(
        responses::received_responses_requests(&source)
            .await
            .is_empty()
    );
    assert_eq!(generation.requests().len(), 3);
    let requests = destination.received_requests().await.unwrap();
    let count_bodies = requests
        .iter()
        .filter(|request| request.url.path().ends_with("/input_tokens"))
        .map(|request| request.body_json::<serde_json::Value>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(count_bodies.len(), 3);
    let tool_outputs = count_bodies[1]["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "function_call_output" && item["call_id"] == "plan-once")
        .collect::<Vec<_>>();
    assert_eq!(tool_outputs.len(), 1);
    assert_eq!(tool_outputs[0]["output"], "Plan updated");
    assert!(
        count_bodies[2]["input"]
            .to_string()
            .contains("The plan was already recorded.")
    );
    assert_eq!(
        count_bodies[2]["input"],
        generation.requests()[2].body_json()["input"]
    );
}

fn destination_context_error() -> wiremock::ResponseTemplate {
    wiremock::ResponseTemplate::new(400).set_body_json(serde_json::json!({
        "error": {
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "message": "The destination tokenizer reports that the prompt exceeds its context."
        }
    }))
}

#[test_case::test_case(false; "destination tokenizer differs")]
#[test_case::test_case(true; "new input crosses destination window")]
#[tokio::test]
async fn actual_destination_overflow_compacts_source_and_retries_same_turn(long_input: bool) {
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    let source_requests = responses::mount_sse_once(
        &source,
        responses::sse(vec![
            responses::ev_assistant_message("summary", "Preserved the original requirement."),
            responses::ev_completed("source-compaction"),
        ]),
    )
    .await;
    let destination_requests = responses::mount_response_sequence(
        &destination,
        vec![
            destination_context_error(),
            wiremock::ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(responses::sse(vec![
                    responses::ev_assistant_message("answer", "Completed the requested work."),
                    responses::ev_completed("destination-answer"),
                ])),
        ],
    )
    .await;
    let (session, turn, _) = transition_test_session(source.uri(), destination.uri()).await;
    use_below_destination_budget(&session).await;
    let input = if long_input {
        "New request details. ".repeat(6_000)
    } else {
        "Continue the original requirement with this new detail.".to_string()
    };
    let result = run_turn(
        session.clone(),
        turn.clone(),
        transition_input(input.clone()),
        &mut McpStartupRequirements::default(),
        None,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    assert_eq!(result.as_deref(), Some("Completed the requested work."));
    let source_calls = source_requests.requests();
    assert_eq!(source_calls.len(), 1);
    assert_eq!(source_calls[0].body_json()["model"], "source-model");
    assert!(source_calls[0].message_input_texts("user").contains(&input));
    let target_calls = destination_requests.requests();
    assert_eq!(target_calls.len(), 2);
    for request in &target_calls {
        assert_eq!(request.body_json()["model"], "destination-model");
        assert_eq!(
            request
                .message_input_texts("user")
                .iter()
                .filter(|text| *text == &input)
                .count(),
            1
        );
    }
    assert!(
        target_calls[1]
            .message_input_texts("user")
            .iter()
            .any(|text| { text.contains("Preserved the original requirement.") })
    );
    let previous = session.previous_turn_settings().await.unwrap();
    assert_eq!(previous.model, "destination-model");
    assert_eq!(previous.model_runtime, Some(turn.model_runtime()));
}

#[test_case::test_case(true; "pre-sampling compaction already ran")]
#[test_case::test_case(false; "destination rejects compacted context")]
#[tokio::test]
async fn destination_overflow_never_repeats_transition_compaction(compact_before_sampling: bool) {
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    let source_requests = responses::mount_sse_once(
        &source,
        responses::sse(vec![
            responses::ev_assistant_message("summary", "The original requirement is preserved."),
            responses::ev_completed("source-compaction"),
        ]),
    )
    .await;
    let expected_destination_requests = if compact_before_sampling { 1 } else { 2 };
    let destination_requests = responses::mount_response_sequence(
        &destination,
        vec![destination_context_error(); expected_destination_requests],
    )
    .await;
    let (session, turn, _) = transition_test_session(source.uri(), destination.uri()).await;
    if !compact_before_sampling {
        use_below_destination_budget(&session).await;
    }
    let previous = session.previous_turn_settings().await;
    let result = run_turn(
        session.clone(),
        turn,
        transition_input("Keep this user request.".to_string()),
        &mut McpStartupRequirements::default(),
        None,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    assert_eq!(result, None);
    assert_eq!(source_requests.requests().len(), 1);
    assert_eq!(
        destination_requests.requests().len(),
        expected_destination_requests
    );
    assert_eq!(session.previous_turn_settings().await, previous);
}

#[test_case::test_case("context_length_exceeded"; "source context limit")]
#[test_case::test_case("server_error"; "source stream failure")]
#[tokio::test]
async fn failed_overflow_compaction_preserves_source_and_does_not_retry_destination(
    source_error: &str,
) {
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    let source_requests = responses::mount_sse_once(
        &source,
        responses::sse_failed("source-failed", source_error, "Source cannot compact."),
    )
    .await;
    let destination_requests =
        responses::mount_response_once(&destination, destination_context_error()).await;
    let (session, turn, _) = transition_test_session(source.uri(), destination.uri()).await;
    use_below_destination_budget(&session).await;
    let previous = session.previous_turn_settings().await;
    let result = run_turn(
        session.clone(),
        turn,
        transition_input("Keep this user request.".to_string()),
        &mut McpStartupRequirements::default(),
        None,
        CancellationToken::new(),
    )
    .await;
    let error = result.unwrap_err();
    assert_eq!(
        matches!(error.details(), CodexErrorDetails::ContextWindowExceeded),
        source_error == "context_length_exceeded"
    );
    assert_eq!(source_requests.requests().len(), 1);
    assert_eq!(destination_requests.requests().len(), 1);
    assert_eq!(session.previous_turn_settings().await, previous);
    let history = session.clone_history().await;
    assert!(history.raw_items().any(|item| {
        matches!(item, ResponseItem::Message { content, .. } if content.iter().any(|part| {
            matches!(part, ContentItem::InputText { text } if text == "Keep the original requirement.")
        }))
    }));
    assert!(history.raw_items().any(|item| {
        matches!(item, ResponseItem::Message { content, .. } if content.iter().any(|part| {
            matches!(part, ContentItem::InputText { text } if text == "Keep this user request.")
        }))
    }));
}

#[test_case::test_case(false; "partial assistant output")]
#[test_case::test_case(true; "tool call already executed")]
#[tokio::test]
async fn overflow_after_model_output_never_replays_the_request(tool_call: bool) {
    use core_test_support::responses;
    let source = responses::start_mock_server().await;
    let destination = responses::start_mock_server().await;
    let mut events = vec![responses::ev_response_created("partial-destination")];
    if tool_call {
        events.push(responses::ev_function_call(
            "plan-once",
            "update_plan",
            r#"{"plan":[{"step":"Record the observed progress","status":"completed"}]}"#,
        ));
    } else {
        events.push(responses::ev_message_item_added("partial-message", ""));
        events.push(responses::ev_output_text_delta(
            "Already started answering.",
        ));
    }
    events.push(serde_json::json!({
        "type": "response.failed",
        "response": {
            "id": "partial-destination",
            "error": {"code": "context_length_exceeded", "message": "Context overflow after progress."}
        }
    }));
    let destination_requests =
        responses::mount_sse_once(&destination, responses::sse(events)).await;
    let (session, turn, _) = transition_test_session(source.uri(), destination.uri()).await;
    use_below_destination_budget(&session).await;
    let result = run_turn(
        session.clone(),
        turn.clone(),
        transition_input("Start this request once.".to_string()),
        &mut McpStartupRequirements::default(),
        None,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    assert_eq!(result, None);
    assert_eq!(destination_requests.requests().len(), 1);
    assert!(
        responses::received_responses_requests(&source)
            .await
            .is_empty()
    );
    assert_eq!(
        session
            .previous_turn_settings()
            .await
            .unwrap()
            .model_runtime,
        Some(turn.model_runtime())
    );
    if tool_call {
        let history = session.clone_history().await;
        let outputs = history
            .raw_items()
            .filter_map(|item| match item {
                ResponseItem::FunctionCallOutput {
                    call_id, output, ..
                } if call_id.as_deref() == Some("plan-once") => Some(output),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].text_content(), Some("Plan updated"));
    }
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
