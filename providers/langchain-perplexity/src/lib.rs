//! Placeholder Perplexity integration crate.
//! Mirrors `.ref/langchain/libs/partners/perplexity` from the Python reference monorepo.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrationDescriptor {
    pub provider: &'static str,
    pub reference_path: &'static str,
}

pub const INTEGRATION: IntegrationDescriptor = IntegrationDescriptor {
    provider: "perplexity",
    reference_path: "libs/partners/perplexity",
};

pub fn integration_descriptor() -> IntegrationDescriptor {
    INTEGRATION
}
