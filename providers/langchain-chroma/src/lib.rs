//! Placeholder Chroma integration crate.
//! Mirrors `.ref/langchain/libs/partners/chroma` from the Python reference monorepo.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrationDescriptor {
    pub provider: &'static str,
    pub reference_path: &'static str,
}

pub const INTEGRATION: IntegrationDescriptor = IntegrationDescriptor {
    provider: "chroma",
    reference_path: "libs/partners/chroma",
};

pub fn integration_descriptor() -> IntegrationDescriptor {
    INTEGRATION
}
