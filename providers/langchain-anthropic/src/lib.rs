//! Placeholder Anthropic integration crate.
//! Mirrors `.ref/langchain/libs/partners/anthropic` from the Python reference monorepo.

/// Describes the current migration status of this provider crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrationDescriptor {
    pub provider: &'static str,
    pub reference_path: &'static str,
}

pub const INTEGRATION: IntegrationDescriptor = IntegrationDescriptor {
    provider: "anthropic",
    reference_path: "libs/partners/anthropic",
};

pub fn integration_descriptor() -> IntegrationDescriptor {
    INTEGRATION
}
