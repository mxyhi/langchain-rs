//! Placeholder Fireworks integration crate.
//! Mirrors `.ref/langchain/libs/partners/fireworks` from the Python reference monorepo.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrationDescriptor {
    pub provider: &'static str,
    pub reference_path: &'static str,
}

pub const INTEGRATION: IntegrationDescriptor = IntegrationDescriptor {
    provider: "fireworks",
    reference_path: "libs/partners/fireworks",
};

pub fn integration_descriptor() -> IntegrationDescriptor {
    INTEGRATION
}
