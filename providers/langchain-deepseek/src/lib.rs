//! Placeholder DeepSeek integration crate.
//! Mirrors `.ref/langchain/libs/partners/deepseek` from the Python reference monorepo.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrationDescriptor {
    pub provider: &'static str,
    pub reference_path: &'static str,
}

pub const INTEGRATION: IntegrationDescriptor = IntegrationDescriptor {
    provider: "deepseek",
    reference_path: "libs/partners/deepseek",
};

pub fn integration_descriptor() -> IntegrationDescriptor {
    INTEGRATION
}
