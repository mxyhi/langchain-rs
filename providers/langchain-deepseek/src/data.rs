pub use langchain_model_profiles::{ProviderCapabilities, ProviderProfile};

pub fn deepseek_profile() -> &'static ProviderProfile {
    langchain_model_profiles::provider("deepseek")
        .expect("deepseek provider profile should exist in langchain-model-profiles")
}

pub fn deepseek_exports() -> &'static [&'static str] {
    deepseek_profile().exports
}

pub fn default_base_url() -> Option<&'static str> {
    deepseek_profile().default_base_url
}
