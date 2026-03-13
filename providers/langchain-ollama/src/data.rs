pub use langchain_model_profiles::{ProviderCapabilities, ProviderProfile};

pub fn ollama_profile() -> &'static ProviderProfile {
    langchain_model_profiles::provider("ollama")
        .expect("ollama provider profile should exist in langchain-model-profiles")
}

pub fn ollama_exports() -> &'static [&'static str] {
    ollama_profile().exports
}

pub fn default_base_url() -> Option<&'static str> {
    ollama_profile().default_base_url
}
