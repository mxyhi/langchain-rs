pub use langchain_model_profiles::{ProviderCapabilities, ProviderProfile};

pub fn perplexity_profile() -> &'static ProviderProfile {
    langchain_model_profiles::provider("perplexity")
        .expect("perplexity provider profile should exist in langchain-model-profiles")
}

pub fn perplexity_exports() -> &'static [&'static str] {
    perplexity_profile().exports
}

pub fn default_base_url() -> Option<&'static str> {
    perplexity_profile().default_base_url
}
