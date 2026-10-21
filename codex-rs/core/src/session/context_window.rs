use super::session::Session;
use super::turn_context::TurnContext;
use crate::config::Config;
use codex_protocol::config_types::AutoCompactTokenLimitScope;
use codex_protocol::openai_models::ModelInfo;

#[derive(Debug)]
pub(crate) struct ContextWindowTokenStatus {
    // Full active context usage, independent of the configured auto-compact scope.
    pub(crate) active_context_tokens: i64,
    // Usage counted against `model_auto_compact_token_limit` for the current scope.
    pub(crate) auto_compact_scope_tokens: i64,
    pub(crate) auto_compact_scope_limit: Option<i64>,
    pub(crate) full_context_window_limit: Option<i64>,
    pub(crate) base_window_tokens_remaining: Option<i64>,
    pub(crate) auto_compact_window_prefill_tokens: Option<i64>,
    pub(crate) full_context_window_limit_reached: bool,
    pub(crate) token_limit_reached: bool,
    pub(crate) turn_end_compaction_threshold_reached: bool,
}

fn tokens_remaining(limit: Option<i64>, used: i64) -> Option<i64> {
    limit.map(|limit| limit.saturating_sub(used).max(0))
}

pub(crate) async fn context_window_token_status(
    sess: &Session,
    turn_context: &TurnContext,
) -> ContextWindowTokenStatus {
    context_window_token_status_with_config(
        sess,
        turn_context.config.as_ref(),
        turn_context.model_info().as_ref(),
    )
    .await
}

pub(crate) async fn context_window_token_status_for_model(
    sess: &Session,
    config: &Config,
    turn_context: &TurnContext,
    model_info: &ModelInfo,
) -> ContextWindowTokenStatus {
    let mut config = config.clone();
    config.token_budget = super::token_budget::resolve_token_budget(
        turn_context.configured_token_budget.as_ref(),
        turn_context.use_model_token_budget_defaults,
        model_info,
    );
    context_window_token_status_with_config(sess, &config, model_info).await
}

/// Evaluates a full prepared input count without recording it as sampled usage.
/// `use_prefill_baseline` is valid only when the saved prefix belongs to this runtime.
/// A model/provider transition has no measured target prefix yet: preserve the native
/// zero body-growth baseline in that case, while checking the complete input's hard limit.
pub(crate) async fn context_window_token_status_with_input_tokens(
    sess: &Session,
    turn_context: &TurnContext,
    model_info: &ModelInfo,
    input_tokens: i64,
    use_prefill_baseline: bool,
) -> ContextWindowTokenStatus {
    let mut config = turn_context.config.as_ref().clone();
    config.token_budget = super::token_budget::resolve_token_budget(
        turn_context.configured_token_budget.as_ref(),
        turn_context.use_model_token_budget_defaults,
        model_info,
    );
    let prefill_input_tokens = if use_prefill_baseline
        && matches!(
            config.model_auto_compact_token_limit_scope,
            AutoCompactTokenLimitScope::BodyAfterPrefix
        ) {
        sess.auto_compact_window_snapshot()
            .await
            .prefill_input_tokens
    } else {
        None
    };
    context_window_token_status_from_input_tokens(
        &config,
        model_info,
        input_tokens,
        prefill_input_tokens,
    )
}

async fn context_window_token_status_with_config(
    sess: &Session,
    config: &Config,
    model_info: &ModelInfo,
) -> ContextWindowTokenStatus {
    let active_context_tokens = sess.get_total_token_usage().await;
    let prefill_input_tokens = if matches!(
        config.model_auto_compact_token_limit_scope,
        AutoCompactTokenLimitScope::BodyAfterPrefix
    ) {
        sess.auto_compact_window_snapshot()
            .await
            .prefill_input_tokens
    } else {
        None
    };
    context_window_token_status_from_input_tokens(
        config,
        model_info,
        active_context_tokens,
        prefill_input_tokens,
    )
}

fn context_window_token_status_from_input_tokens(
    config: &Config,
    model_info: &ModelInfo,
    active_context_tokens: i64,
    prefill_input_tokens: Option<i64>,
) -> ContextWindowTokenStatus {
    // Count either the full active context or only the tokens added after the initial prefix.
    let (auto_compact_scope_tokens, auto_compact_scope_limit, auto_compact_window_prefill_tokens) =
        match config.model_auto_compact_token_limit_scope {
            AutoCompactTokenLimitScope::Total => (
                active_context_tokens,
                model_info.auto_compact_token_limit(),
                None,
            ),
            AutoCompactTokenLimitScope::BodyAfterPrefix => {
                let baseline = prefill_input_tokens.unwrap_or(active_context_tokens);

                let scope_limit = config
                    .model_auto_compact_token_limit
                    .or_else(|| model_info.auto_compact_token_limit());
                (
                    active_context_tokens.saturating_sub(baseline),
                    scope_limit,
                    prefill_input_tokens,
                )
            }
        };

    // The model's full context window is a hard cap, independent of the auto-compaction scope.
    let full_context_window_limit = model_info.resolved_context_window().map(|context_window| {
        context_window.saturating_mul(model_info.effective_context_window_percent) / 100
    });

    // Report remaining tokens against the base (unbuffered) window, capped by the full context.
    let base_window_tokens_remaining = [
        tokens_remaining(auto_compact_scope_limit, auto_compact_scope_tokens),
        tokens_remaining(full_context_window_limit, active_context_tokens),
    ]
    .into_iter()
    .flatten()
    .min();

    // Only reserve the fallback buffer when there is a fallback prompt to use it.
    let auto_compact_fallback_buffer_tokens = config
        .token_budget
        .as_ref()
        .map_or(0, crate::config::TokenBudgetConfig::fallback_buffer_tokens);
    let buffered_auto_compact_limit = auto_compact_scope_limit
        .map(|limit| limit.saturating_add(auto_compact_fallback_buffer_tokens));

    // Force compaction once the buffered window or the model's full context window is reached.
    let full_context_window_limit_reached =
        full_context_window_limit.is_some_and(|limit| active_context_tokens >= limit);
    let token_limit_reached = buffered_auto_compact_limit
        .is_some_and(|limit| auto_compact_scope_tokens >= limit)
        || full_context_window_limit_reached;
    let post_turn_percent = config.model_post_turn_compact_threshold_percent;
    let turn_end_compaction_threshold_reached = post_turn_percent > 0
        && (token_limit_reached
            || full_context_window_limit.is_some_and(|limit| {
                i128::from(active_context_tokens) * 100
                    >= i128::from(limit) * i128::from(post_turn_percent)
            }));

    ContextWindowTokenStatus {
        active_context_tokens,
        auto_compact_scope_tokens,
        auto_compact_scope_limit,
        full_context_window_limit,
        base_window_tokens_remaining,
        auto_compact_window_prefill_tokens,
        full_context_window_limit_reached,
        token_limit_reached,
        turn_end_compaction_threshold_reached,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TokenBudgetConfig;
    use crate::session::tests::make_session_and_context;
    use codex_protocol::protocol::TokenUsage;
    use codex_protocol::protocol::TokenUsageInfo;
    use pretty_assertions::assert_eq;
    use std::sync::Arc;

    async fn counted_input_fixture(
        scope: AutoCompactTokenLimitScope,
    ) -> (Session, TurnContext, ModelInfo) {
        let (sess, mut turn) = make_session_and_context().await;
        let config = Arc::make_mut(&mut turn.config);
        config.model_auto_compact_token_limit_scope = scope;
        config.model_auto_compact_token_limit = Some(200);
        config.model_post_turn_compact_threshold_percent = 90;
        config.token_budget = None;
        turn.configured_token_budget = None;
        turn.use_model_token_budget_defaults = false;
        let mut model = turn.model_info().as_ref().clone();
        model.slug = "counted-target-model".to_owned();
        model.context_window = Some(1_000);
        model.auto_compact_token_limit = Some(800);
        model.effective_context_window_percent = 90;
        let usage = TokenUsage {
            input_tokens: 9_000,
            output_tokens: 500,
            total_tokens: 9_500,
            ..Default::default()
        };
        {
            let mut state = sess.state.lock().await;
            state.set_token_info(Some(TokenUsageInfo {
                total_token_usage: usage.clone(),
                last_token_usage: usage,
                model_context_window: Some(32_000),
            }));
            state.ensure_auto_compact_window_server_prefill_from_usage(&TokenUsage {
                input_tokens: 600,
                total_tokens: 600,
                ..Default::default()
            });
        }
        (sess, turn, model)
    }

    #[tokio::test]
    async fn exact_input_count_uses_step_model_budget_without_mutating_stored_usage() {
        let (sess, mut turn, model) =
            counted_input_fixture(AutoCompactTokenLimitScope::Total).await;
        turn.configured_token_budget = Some(TokenBudgetConfig {
            auto_compact_fallback_prompt: Some("Summarize remaining work.".to_owned()),
            auto_compact_fallback_buffer_tokens: Some(100),
            ..TokenBudgetConfig::default()
        });
        let before_usage = sess.state.lock().await.token_info();
        let before_window = sess.auto_compact_window_snapshot().await;
        for (count, remaining, hard_reached, token_reached, post_turn_reached) in [
            (750, 50, false, false, false),
            (850, 0, false, false, true),
            (900, 0, true, true, true),
        ] {
            let status =
                context_window_token_status_with_input_tokens(&sess, &turn, &model, count, false)
                    .await;
            assert_eq!(status.active_context_tokens, count);
            assert_eq!(status.auto_compact_scope_tokens, count);
            assert_eq!(status.auto_compact_scope_limit, Some(800));
            assert_eq!(status.full_context_window_limit, Some(900));
            assert_eq!(status.base_window_tokens_remaining, Some(remaining));
            assert_eq!(status.auto_compact_window_prefill_tokens, None);
            assert_eq!(status.full_context_window_limit_reached, hard_reached);
            assert_eq!(status.token_limit_reached, token_reached);
            assert_eq!(
                status.turn_end_compaction_threshold_reached,
                post_turn_reached
            );
        }
        assert_eq!(sess.state.lock().await.token_info(), before_usage);
        assert_eq!(sess.auto_compact_window_snapshot().await, before_window);
    }

    #[tokio::test]
    async fn body_after_prefix_exact_count_uses_same_runtime_baseline_without_storing_count() {
        let (sess, turn, model) =
            counted_input_fixture(AutoCompactTokenLimitScope::BodyAfterPrefix).await;
        let before_usage = sess.state.lock().await.token_info();
        let before_window = sess.auto_compact_window_snapshot().await;
        for (count, body, remaining, reached) in [(750, 150, 50, false), (800, 200, 0, true)] {
            let status =
                context_window_token_status_with_input_tokens(&sess, &turn, &model, count, true)
                    .await;
            assert_eq!(status.active_context_tokens, count);
            assert_eq!(status.auto_compact_scope_tokens, body);
            assert_eq!(status.auto_compact_scope_limit, Some(200));
            assert_eq!(status.auto_compact_window_prefill_tokens, Some(600));
            assert_eq!(status.base_window_tokens_remaining, Some(remaining));
            assert!(!status.full_context_window_limit_reached);
            assert_eq!(status.token_limit_reached, reached);
        }
        assert_eq!(sess.state.lock().await.token_info(), before_usage);
        assert_eq!(sess.auto_compact_window_snapshot().await, before_window);
    }

    #[tokio::test]
    async fn body_after_prefix_transition_has_zero_unknown_body_but_checks_full_input_limit() {
        let (sess, turn, model) =
            counted_input_fixture(AutoCompactTokenLimitScope::BodyAfterPrefix).await;
        let before_usage = sess.state.lock().await.token_info();
        let before_window = sess.auto_compact_window_snapshot().await;
        // The source's prefix count cannot measure the target's body growth. A missing
        // target baseline keeps native zero growth, while its complete input is checked.
        for (count, remaining, reached) in [(850, 50, false), (900, 0, true)] {
            let status =
                context_window_token_status_with_input_tokens(&sess, &turn, &model, count, false)
                    .await;
            assert_eq!(status.active_context_tokens, count);
            assert_eq!(status.auto_compact_scope_tokens, 0);
            assert_eq!(status.auto_compact_scope_limit, Some(200));
            assert_eq!(status.auto_compact_window_prefill_tokens, None);
            assert_eq!(status.full_context_window_limit, Some(900));
            assert_eq!(status.base_window_tokens_remaining, Some(remaining));
            assert_eq!(status.full_context_window_limit_reached, reached);
            assert_eq!(status.token_limit_reached, reached);
        }
        assert_eq!(sess.state.lock().await.token_info(), before_usage);
        assert_eq!(sess.auto_compact_window_snapshot().await, before_window);
    }
}
