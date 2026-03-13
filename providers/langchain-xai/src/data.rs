pub use langchain_model_profiles::{ProviderCapabilities, ProviderProfile};

pub fn xai_profile() -> &'static ProviderProfile {
    langchain_model_profiles::provider("xai")
        .expect("xai provider profile should exist in langchain-model-profiles")
}

pub fn xai_exports() -> &'static [&'static str] {
    xai_profile().exports
}

pub fn default_base_url() -> Option<&'static str> {
    xai_profile().default_base_url
}
