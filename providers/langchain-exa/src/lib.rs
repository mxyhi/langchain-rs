//! Placeholder Exa integration crate.
//! Mirrors `.ref/langchain/libs/partners/exa` from the Python reference monorepo.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrationDescriptor {
    pub provider: &'static str,
    pub reference_path: &'static str,
}

pub const INTEGRATION: IntegrationDescriptor = IntegrationDescriptor {
    provider: "exa",
    reference_path: "libs/partners/exa",
};

pub fn integration_descriptor() -> IntegrationDescriptor {
    INTEGRATION
}
