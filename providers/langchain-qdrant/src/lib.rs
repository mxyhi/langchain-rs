//! Placeholder Qdrant integration crate.
//! Mirrors `.ref/langchain/libs/partners/qdrant` from the Python reference monorepo.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrationDescriptor {
    pub provider: &'static str,
    pub reference_path: &'static str,
}

pub const INTEGRATION: IntegrationDescriptor = IntegrationDescriptor {
    provider: "qdrant",
    reference_path: "libs/partners/qdrant",
};

pub fn integration_descriptor() -> IntegrationDescriptor {
    INTEGRATION
}
