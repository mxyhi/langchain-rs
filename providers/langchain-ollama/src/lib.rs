//! Placeholder Ollama integration crate.
//! Mirrors `.ref/langchain/libs/partners/ollama` from the Python reference monorepo.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrationDescriptor {
    pub provider: &'static str,
    pub reference_path: &'static str,
}

pub const INTEGRATION: IntegrationDescriptor = IntegrationDescriptor {
    provider: "ollama",
    reference_path: "libs/partners/ollama",
};

pub fn integration_descriptor() -> IntegrationDescriptor {
    INTEGRATION
}
