use super::*;
use crate::create_model_provider;
use codex_model_provider_info::ModelProviderCapabilities;
use pretty_assertions::assert_eq;

#[test]
fn capability_overrides_preserve_unspecified_provider_defaults() {
    for info in [
        ModelProviderInfo::default(),
        ModelProviderInfo::create_openai_provider(/*base_url*/ None),
        ModelProviderInfo {
            name: "Azure".to_string(),
            ..Default::default()
        },
    ] {
        let defaults = create_model_provider(info.clone(), /*auth_manager*/ None).capabilities();
        let provider = create_model_provider(
            ModelProviderInfo {
                capabilities: Some(ModelProviderCapabilities {
                    external_web_access: Some(false),
                    ..Default::default()
                }),
                ..info
            },
            /*auth_manager*/ None,
        );
        assert_eq!(
            provider.capabilities(),
            ProviderCapabilities {
                external_web_access: false,
                ..defaults
            },
        );
    }
}

#[test]
fn encrypted_content_capability_can_be_declared_by_custom_providers() {
    for supported in [false, true] {
        let provider = create_model_provider(
            ModelProviderInfo {
                name: "Configured service".into(),
                capabilities: Some(ModelProviderCapabilities {
                    encrypted_content: Some(supported),
                    ..Default::default()
                }),
                ..Default::default()
            },
            None,
        );
        assert_eq!(provider.capabilities().encrypted_content, supported);
    }
}

#[test]
fn agent_messages_default_to_native_openai_and_override_independently() {
    for (info, default_native) in [
        (ModelProviderInfo::default(), false),
        (ModelProviderInfo::create_openai_provider(None), true),
        (
            ModelProviderInfo {
                name: "Azure".into(),
                ..Default::default()
            },
            false,
        ),
    ] {
        assert_eq!(
            create_model_provider(info.clone(), None)
                .capabilities()
                .agent_messages,
            default_native
        );
        for native in [false, true] {
            for encrypted in [false, true] {
                let provider = create_model_provider(
                    ModelProviderInfo {
                        capabilities: Some(ModelProviderCapabilities {
                            agent_messages: Some(native),
                            encrypted_content: Some(encrypted),
                            ..Default::default()
                        }),
                        ..info.clone()
                    },
                    None,
                );
                assert_eq!(provider.capabilities().agent_messages, native);
                assert_eq!(provider.capabilities().encrypted_content, encrypted);
            }
        }
    }
}
