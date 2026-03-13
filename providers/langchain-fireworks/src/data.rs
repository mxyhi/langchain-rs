pub use langchain_model_profiles::{ProviderCapabilities, ProviderProfile};

pub fn fireworks_profile() -> &'static ProviderProfile {
    langchain_model_profiles::provider("fireworks")
        .expect("fireworks provider profile should exist in langchain-model-profiles")
}

pub fn fireworks_exports() -> &'static [&'static str] {
    fireworks_profile().exports
}

pub fn default_base_url() -> Option<&'static str> {
    fireworks_profile().default_base_url
}
