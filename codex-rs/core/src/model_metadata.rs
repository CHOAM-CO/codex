//! Resolve API model metadata separately from the external harness catalog.

use crate::config::Config;
use codex_models_manager::ModelsManagerConfig;
use codex_models_manager::manager::ModelsManager;
use codex_models_manager::model_info::host_tool_model_info;
use codex_models_manager::model_info::with_config_overrides;
use codex_protocol::external_executor::ModelExecutor;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ReasoningEffortPreset;

pub(crate) async fn resolve_model_info(
    config: &Config,
    models_manager: &dyn ModelsManager,
    model: &str,
    overrides: &ModelsManagerConfig,
) -> ModelInfo {
    match config.model_executor {
        ModelExecutor::Codex => models_manager.get_model_info(model, overrides).await,
        ModelExecutor::Antigravity => {
            let mut info = host_tool_model_info(model);
            if let Some(metadata) = config
                .external_model_catalog
                .iter()
                .find(|metadata| metadata.model == model)
            {
                info.display_name = metadata.display_name.clone();
                info.multi_agent_reasoning_effort = metadata.multi_agent_reasoning_effort.clone();
                info.supported_reasoning_levels = metadata
                    .supported_reasoning_efforts
                    .iter()
                    .map(|effort| ReasoningEffortPreset {
                        effort: effort.clone(),
                        description: effort.to_string(),
                    })
                    .collect();
            }
            with_config_overrides(info, overrides)
        }
    }
}
