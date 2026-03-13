pub use langchain_model_profiles::{ProviderCapabilities, ProviderProfile};

pub fn huggingface_profile() -> &'static ProviderProfile {
    langchain_model_profiles::provider("huggingface")
        .expect("huggingface provider profile should exist in langchain-model-profiles")
}

pub fn huggingface_exports() -> &'static [&'static str] {
    huggingface_profile().exports
}

pub fn default_base_url() -> Option<&'static str> {
    huggingface_profile().default_base_url
}
