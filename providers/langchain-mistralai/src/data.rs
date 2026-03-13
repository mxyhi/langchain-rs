pub use langchain_model_profiles::{ProviderCapabilities, ProviderProfile};

pub fn mistralai_profile() -> &'static ProviderProfile {
    langchain_model_profiles::provider("mistralai")
        .expect("mistralai provider profile should exist in langchain-model-profiles")
}

pub fn mistralai_exports() -> &'static [&'static str] {
    mistralai_profile().exports
}

pub fn default_base_url() -> Option<&'static str> {
    mistralai_profile().default_base_url
}
