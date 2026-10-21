//! Serializable capability overrides for custom Responses-compatible providers.
//!
//! Unspecified capabilities inherit the provider's existing defaults. Model and
//! turn settings may further restrict the capabilities declared here.

use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

/// Overrides for the optional API features supported by a custom provider.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct ModelProviderCapabilities {
    /// Whether hosted web search may access the live web.
    pub external_web_access: Option<bool>,
    /// Accepts encrypted Responses history and tool-content items.
    /// Override for a provider that can replay the source ciphertext.
    pub encrypted_content: Option<bool>,
    /// Accepts native `agent_message` Responses input items.
    /// Otherwise inter-agent input is sent as an ordinary message envelope.
    pub agent_messages: Option<bool>,
    /// Remote context-compaction protocol; omission preserves provider defaults.
    pub remote_compaction: Option<RemoteCompactionSupport>,
}

/// Remote context-compaction protocols supported by a model provider.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteCompactionSupport {
    /// Use local compaction instead of a provider compaction protocol.
    Unsupported,
    /// Send `compaction_trigger` items over the Responses endpoint.
    V2,
}
