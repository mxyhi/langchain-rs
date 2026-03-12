//! Placeholder Mistral AI integration crate.
//! Mirrors `.ref/langchain/libs/partners/mistralai` from the Python reference monorepo.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrationDescriptor {
    pub provider: &'static str,
    pub reference_path: &'static str,
}

pub const INTEGRATION: IntegrationDescriptor = IntegrationDescriptor {
    provider: "mistralai",
    reference_path: "libs/partners/mistralai",
};

pub fn integration_descriptor() -> IntegrationDescriptor {
    INTEGRATION
}
