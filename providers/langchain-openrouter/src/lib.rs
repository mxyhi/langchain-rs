//! Placeholder OpenRouter integration crate.
//! Mirrors `.ref/langchain/libs/partners/openrouter` from the Python reference monorepo.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrationDescriptor {
    pub provider: &'static str,
    pub reference_path: &'static str,
}

pub const INTEGRATION: IntegrationDescriptor = IntegrationDescriptor {
    provider: "openrouter",
    reference_path: "libs/partners/openrouter",
};

pub fn integration_descriptor() -> IntegrationDescriptor {
    INTEGRATION
}
