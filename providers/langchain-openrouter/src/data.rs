pub use langchain_model_profiles::{ProviderCapabilities, ProviderProfile};

pub fn openrouter_profile() -> &'static ProviderProfile {
    langchain_model_profiles::provider("openrouter")
        .expect("openrouter provider profile should exist in langchain-model-profiles")
}

pub fn openrouter_exports() -> &'static [&'static str] {
    openrouter_profile().exports
}

pub fn default_base_url() -> Option<&'static str> {
    openrouter_profile().default_base_url
}
