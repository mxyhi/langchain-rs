//! Placeholder Hugging Face integration crate.
//! Mirrors `.ref/langchain/libs/partners/huggingface` from the Python reference monorepo.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrationDescriptor {
    pub provider: &'static str,
    pub reference_path: &'static str,
}

pub const INTEGRATION: IntegrationDescriptor = IntegrationDescriptor {
    provider: "huggingface",
    reference_path: "libs/partners/huggingface",
};

pub fn integration_descriptor() -> IntegrationDescriptor {
    INTEGRATION
}
