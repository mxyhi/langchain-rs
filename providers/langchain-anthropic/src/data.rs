pub use langchain_model_profiles::{ProviderCapabilities, ProviderProfile};

pub fn anthropic_profile() -> &'static ProviderProfile {
    langchain_model_profiles::provider("anthropic")
        .expect("anthropic provider profile should exist in langchain-model-profiles")
}

pub fn anthropic_exports() -> &'static [&'static str] {
    anthropic_profile().exports
}

pub fn default_base_url() -> Option<&'static str> {
    anthropic_profile().default_base_url
}
