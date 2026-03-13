pub use langchain_model_profiles::{ProviderCapabilities, ProviderProfile};

pub fn openai_profile() -> &'static ProviderProfile {
    langchain_model_profiles::provider("openai")
        .expect("openai provider profile should exist in langchain-model-profiles")
}

pub fn openai_exports() -> &'static [&'static str] {
    openai_profile().exports
}

pub fn default_base_url() -> Option<&'static str> {
    openai_profile().default_base_url
}
