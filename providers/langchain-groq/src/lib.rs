//! Placeholder Groq integration crate.
//! Mirrors `.ref/langchain/libs/partners/groq` from the Python reference monorepo.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrationDescriptor {
    pub provider: &'static str,
    pub reference_path: &'static str,
}

pub const INTEGRATION: IntegrationDescriptor = IntegrationDescriptor {
    provider: "groq",
    reference_path: "libs/partners/groq",
};

pub fn integration_descriptor() -> IntegrationDescriptor {
    INTEGRATION
}
