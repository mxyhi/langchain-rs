pub use langchain_model_profiles::{ProviderCapabilities, ProviderProfile};

pub fn groq_profile() -> &'static ProviderProfile {
    langchain_model_profiles::provider("groq")
        .expect("groq provider profile should exist in langchain-model-profiles")
}

pub fn groq_exports() -> &'static [&'static str] {
    groq_profile().exports
}

pub fn default_base_url() -> Option<&'static str> {
    groq_profile().default_base_url
}
