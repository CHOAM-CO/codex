use super::*;
use crate::config::AgentRoleConfig;
use crate::session::tests::make_session_and_context;
use crate::session::tests::make_session_and_context_with_auth_and_config_and_rx;
use codex_login::CodexAuth;
use codex_model_provider::create_model_provider;

fn options<'a>(version: SpawnConfigVersion, full_history_fork: bool) -> SpawnConfigOptions<'a> {
    SpawnConfigOptions {
        version,
        full_history_fork,
        role_name: None,
        model: None,
        model_provider: None,
        executor: None,
        reasoning_effort: None,
    }
}

#[tokio::test]
async fn configured_child_provider_is_applied_to_fresh_and_full_history_children() {
    for version in [SpawnConfigVersion::V1, SpawnConfigVersion::V2] {
        for full_history in [false, true] {
            let (session, mut turn) = make_session_and_context().await;
            let config = Arc::make_mut(&mut turn.config);
            config.agent_default_subagent_model = Some("local-research-model".into());
            config.agent_default_subagent_model_provider = Some("ollama".into());
            let step = StepContext::for_test(Arc::new(turn));
            let prepared =
                prepare_agent_spawn_config(&session, &step, options(version, full_history))
                    .await
                    .unwrap();
            assert_eq!(
                prepared.config.model.as_deref(),
                Some("local-research-model")
            );
            assert_eq!(prepared.config.model_provider_id, "ollama");
            assert_eq!(
                prepared.config.model_provider,
                prepared.config.model_providers["ollama"]
            );
        }
    }
}

#[tokio::test]
async fn explicit_child_selection_does_not_mix_with_configured_default_pair() {
    let (session, mut turn) = make_session_and_context().await;
    let parent_model = turn.model_info().slug.clone();
    let parent_provider = turn.config.model_provider_id.clone();
    let config = Arc::make_mut(&mut turn.config);
    config.agent_default_subagent_model = Some("local-default".into());
    config.agent_default_subagent_model_provider = Some("ollama".into());
    let step = StepContext::for_test(Arc::new(turn));
    for (model, executor) in [
        (Some(parent_model.as_str()), None),
        (None, Some(ModelExecutor::Antigravity)),
    ] {
        let mut request = options(SpawnConfigVersion::V2, false);
        request.model = model;
        request.executor = executor;
        let prepared = prepare_agent_spawn_config(&session, &step, request)
            .await
            .unwrap();
        assert_eq!(
            prepared.config.model.as_deref(),
            Some(parent_model.as_str())
        );
        assert_eq!(prepared.config.model_provider_id, parent_provider);
        assert_eq!(
            prepared.config.model_executor,
            executor.unwrap_or(step.turn.config.model_executor)
        );
        assert_eq!(
            step.turn.config.agent_default_subagent_model.as_deref(),
            Some("local-default")
        );
        assert_eq!(
            step.turn
                .config
                .agent_default_subagent_model_provider
                .as_deref(),
            Some("ollama")
        );
    }
}

#[tokio::test]
async fn openai_child_uses_openai_metadata_under_a_local_parent() {
    let (session, turn, _events) = make_session_and_context_with_auth_and_config_and_rx(
        CodexAuth::from_api_key("test-openai-key"),
        Vec::new(),
        |config| {
            config.model = Some("gpt-6-luna".into());
            config.model_provider_id = "ollama".into();
            config
                .model_providers
                .get_mut("ollama")
                .unwrap()
                .experimental_bearer_token = Some("test-local-key".into());
            config.model_provider = config.model_providers["ollama"].clone();
            config.agent_default_subagent_model = Some("configured-other-model".into());
            config.agent_default_subagent_model_provider = Some("openai".into());
        },
    )
    .await;
    assert_eq!(
        session.services.model_client.provider_info(),
        &turn.config.model_providers["ollama"]
    );
    assert_eq!(
        turn.provider.api_auth().await.unwrap().to_auth_headers()[http::header::AUTHORIZATION],
        "Bearer test-local-key",
        "the local parent uses its own configured credentials"
    );
    assert!(Arc::ptr_eq(
        &spawn_agent_models_manager(&session, &turn.config),
        &session.services.models_manager
    ));
    let step = StepContext::for_test(turn);
    let selection = |config: &Config| {
        (
            config.model.clone(),
            config.model_provider_id.clone(),
            config.model_provider.clone(),
            config.model_executor,
            config.agent_default_subagent_model.clone(),
            config.agent_default_subagent_model_provider.clone(),
            config.agent_default_subagent_executor,
        )
    };
    let parent_before = selection(&step.turn.config);
    let providers_before = step.turn.config.model_providers.clone();
    let mut sibling_request = options(SpawnConfigVersion::V2, false);
    sibling_request.executor = Some(ModelExecutor::Antigravity);
    sibling_request.model = Some("external-sibling");
    let sibling = prepare_agent_spawn_config(&session, &step, sibling_request)
        .await
        .unwrap();
    let sibling_before = selection(&sibling.config);
    // The same model ID under another provider must still build that provider's manager.
    for model in [Some("gpt-6-luna"), None] {
        let mut request = options(SpawnConfigVersion::V2, true);
        request.model_provider = Some("openai");
        request.model = model;
        let prepared = prepare_agent_spawn_config(&session, &step, request)
            .await
            .unwrap();
        assert_eq!(prepared.config.model.as_deref(), Some("gpt-6-luna"));
        assert_eq!(prepared.config.model_provider_id, "openai");
        assert_eq!(
            prepared.config.model_provider,
            prepared.config.model_providers["openai"]
        );
        let selected_manager = spawn_agent_models_manager(&session, &prepared.config);
        assert!(!Arc::ptr_eq(
            &selected_manager,
            &session.services.models_manager
        ));
        let info = selected_manager
            .get_model_info("gpt-6-luna", &prepared.config.to_models_manager_config())
            .await;
        assert_eq!(info.slug, "gpt-6-luna");
        assert!(!info.used_fallback_model_metadata);
        let selected_provider = create_model_provider(
            prepared.config.model_provider,
            Some(Arc::clone(&session.services.auth_manager)),
        );
        assert!(Arc::ptr_eq(
            &selected_provider.auth_manager().unwrap(),
            &session.services.auth_manager
        ));
        assert_eq!(
            selected_provider
                .api_auth()
                .await
                .unwrap()
                .to_auth_headers()[http::header::AUTHORIZATION],
            "Bearer test-openai-key"
        );
        assert_eq!(selection(&step.turn.config), parent_before);
        assert_eq!(step.turn.config.model_providers, providers_before);
        assert_eq!(selection(&sibling.config), sibling_before);
    }
}

#[tokio::test]
async fn role_provider_and_model_override_tool_selection() {
    let (session, mut turn) = make_session_and_context().await;
    let role_dir = tempfile::tempdir().unwrap();
    let path = role_dir.path().join("reviewer.toml");
    tokio::fs::write(
        &path,
        "model = 'role-local-model'\nmodel_provider = 'ollama'\n",
    )
    .await
    .unwrap();
    Arc::make_mut(&mut turn.config).agent_roles.insert(
        "reviewer".into(),
        AgentRoleConfig {
            description: None,
            config_file: Some(path),
            nickname_candidates: None,
        },
    );
    let step = StepContext::for_test(Arc::new(turn));
    let mut request = options(SpawnConfigVersion::V2, true);
    request.model_provider = Some("openai");
    request.model = Some("gpt-6-luna");
    request.role_name = Some("reviewer");
    let prepared = prepare_agent_spawn_config(&session, &step, request)
        .await
        .unwrap();
    assert_eq!(prepared.config.model.as_deref(), Some("role-local-model"));
    assert_eq!(prepared.config.model_provider_id, "ollama");
}

#[tokio::test]
async fn external_child_preserves_requested_model_without_responses_catalog_validation() {
    let (session, turn) = make_session_and_context().await;
    let step = StepContext::for_test(Arc::new(turn));
    let mut request = options(SpawnConfigVersion::V2, true);
    request.executor = Some(ModelExecutor::Antigravity);
    request.model = Some("external-catalog-model");
    let prepared = prepare_agent_spawn_config(&session, &step, request)
        .await
        .unwrap();
    assert_eq!(prepared.config.model_executor, ModelExecutor::Antigravity);
    assert_eq!(
        prepared.config.model.as_deref(),
        Some("external-catalog-model")
    );
}

#[tokio::test]
async fn unknown_native_provider_is_reported_without_replacement() {
    let (session, turn) = make_session_and_context().await;
    let step = StepContext::for_test(Arc::new(turn));
    let mut request = options(SpawnConfigVersion::V2, false);
    request.model_provider = Some("not-configured");
    let result = prepare_agent_spawn_config(&session, &step, request).await;
    assert!(
        matches!(result, Err(message) if message == "Model provider `not-configured` not found")
    );
}

#[tokio::test]
async fn external_parent_partial_selections_keep_the_inherited_executor() {
    for version in [SpawnConfigVersion::V1, SpawnConfigVersion::V2] {
        for field in ["model", "provider", "configured_default"] {
            let (session, mut turn) = make_session_and_context().await;
            let config = Arc::make_mut(&mut turn.config);
            config.model_executor = ModelExecutor::Antigravity;
            if field == "configured_default" {
                config.agent_default_subagent_model = Some("external-default".into());
            }
            let step = StepContext::for_test(Arc::new(turn));
            let mut request = options(version, false);
            if field == "model" {
                request.model = Some("external-selected");
            } else if field == "provider" {
                request.model_provider = Some("ollama");
            }
            let prepared = prepare_agent_spawn_config(&session, &step, request)
                .await
                .unwrap();
            assert_eq!(prepared.config.model_executor, ModelExecutor::Antigravity);
            if field == "model" {
                assert_eq!(prepared.config.model.as_deref(), Some("external-selected"));
            } else if field == "provider" {
                assert_eq!(prepared.config.model_provider_id, "ollama");
            } else {
                assert_eq!(prepared.config.model.as_deref(), Some("external-default"));
            }
        }
    }
}

#[tokio::test]
async fn partial_role_selection_keeps_external_executor_unless_role_overrides_it() {
    for (role_config, expected) in [
        (
            "model = 'external-reviewer'\nmodel_provider = 'ollama'\n",
            ModelExecutor::Antigravity,
        ),
        (
            "model = 'local-reviewer'\nmodel_provider = 'ollama'\nmodel_executor = 'codex'\n",
            ModelExecutor::Codex,
        ),
    ] {
        let (session, mut turn) = make_session_and_context().await;
        let role_dir = tempfile::tempdir().unwrap();
        let path = role_dir.path().join("reviewer.toml");
        tokio::fs::write(&path, role_config).await.unwrap();
        let config = Arc::make_mut(&mut turn.config);
        config.model_executor = ModelExecutor::Antigravity;
        config.agent_roles.insert(
            "reviewer".into(),
            AgentRoleConfig {
                description: None,
                config_file: Some(path),
                nickname_candidates: None,
            },
        );
        let step = StepContext::for_test(Arc::new(turn));
        let mut request = options(SpawnConfigVersion::V2, true);
        request.model = Some("external-selected");
        request.role_name = Some("reviewer");
        let prepared = prepare_agent_spawn_config(&session, &step, request)
            .await
            .unwrap();
        assert_eq!(prepared.config.model_executor, expected);
        assert_eq!(prepared.config.model_provider_id, "ollama");
    }
}

#[test]
fn missing_saved_child_model_is_reported_without_parent_substitution() {
    let empty = codex_history::InitialHistory::New;
    assert_eq!(
        saved_agent_model(Some("recorded-child-model"), &empty).unwrap(),
        "recorded-child-model"
    );
    assert_eq!(
        saved_agent_model(None, &empty).unwrap_err(),
        "The saved child model is unavailable in thread metadata and owned history"
    );
}

#[tokio::test]
async fn native_selection_errors_preserve_parent_catalog_and_credentials() {
    let (session, turn) = make_session_and_context().await;
    let step = StepContext::for_test(Arc::new(turn));
    let selection = |config: &Config| {
        (
            config.model.clone(),
            config.model_provider_id.clone(),
            config.model_provider.clone(),
            config.model_executor,
            config.model_reasoning_effort.clone(),
        )
    };
    let parent = selection(&step.turn.config);
    let providers = step.turn.config.model_providers.clone();
    let auth = Arc::clone(&session.services.auth_manager);
    let headers = step
        .turn
        .provider
        .api_auth()
        .await
        .unwrap()
        .to_auth_headers();
    let manager = spawn_agent_models_manager(&session, &step.turn.config);
    let available = manager
        .list_models(
            RefreshStrategy::Offline,
            step.turn.config.http_client_factory(),
        )
        .await;
    let available_names = available
        .iter()
        .filter(|preset| preset.show_in_picker)
        .filter(|preset| model_supports_multi_agent_backend(preset, step.turn.multi_agent_version))
        .take(MAX_SPAWN_AGENT_MODEL_OVERRIDES)
        .map(|preset| preset.model.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let info = manager
        .get_model_info("gpt-6-luna", &step.turn.config.to_models_manager_config())
        .await;
    assert!(!info.used_fallback_model_metadata);
    assert!(
        !info
            .supported_reasoning_levels
            .iter()
            .any(|level| level.effort == ReasoningEffort::None)
    );
    let supported = info
        .supported_reasoning_levels
        .iter()
        .map(|level| level.effort.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    for (model, effort, expected) in [
        (
            "not-in-native-catalog",
            None,
            format!(
                "Unknown model `not-in-native-catalog` for spawn_agent. Available models: {available_names}"
            ),
        ),
        (
            "gpt-6-luna",
            Some(ReasoningEffort::None),
            format!(
                "Reasoning effort `none` is not supported for model `gpt-6-luna`. Supported reasoning efforts: {supported}"
            ),
        ),
    ] {
        let mut request = options(SpawnConfigVersion::V2, false);
        request.model = Some(model);
        request.reasoning_effort = effort;
        let result = prepare_agent_spawn_config(&session, &step, request).await;
        let Err(error) = result else {
            panic!("native unsupported selection must report its error")
        };
        assert_eq!(error, expected);
        assert_eq!(selection(&step.turn.config), parent);
        assert_eq!(step.turn.config.model_providers, providers);
        assert!(Arc::ptr_eq(&session.services.auth_manager, &auth));
        assert_eq!(
            step.turn
                .provider
                .api_auth()
                .await
                .unwrap()
                .to_auth_headers(),
            headers
        );
    }
}
