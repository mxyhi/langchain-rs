//! Model capability profile primitives.
//!
//! This crate will host model metadata comparable to Python
//! `langchain-model-profiles`: provider/model capability data that can be shared
//! by integration crates without hard-coding ad-hoc feature flags everywhere.

/// Minimal model capability record placeholder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProfile {
    provider: String,
    model: String,
}

impl ModelProfile {
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

/// Registry placeholder for future capability lookup/loading logic.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelProfileRegistry;

impl ModelProfileRegistry {
    pub const fn package_name(self) -> &'static str {
        "langchain-model-profiles"
    }

    pub const fn purpose(self) -> &'static str {
        "Shared model capability metadata for LangChain Rust integrations."
    }
}
