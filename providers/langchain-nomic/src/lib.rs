//! Placeholder Nomic integration crate.
//! Mirrors `.ref/langchain/libs/partners/nomic` from the Python reference monorepo.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrationDescriptor {
    pub provider: &'static str,
    pub reference_path: &'static str,
}

pub const INTEGRATION: IntegrationDescriptor = IntegrationDescriptor {
    provider: "nomic",
    reference_path: "libs/partners/nomic",
};

pub fn integration_descriptor() -> IntegrationDescriptor {
    INTEGRATION
}
