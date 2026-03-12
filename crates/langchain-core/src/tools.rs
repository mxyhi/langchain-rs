use crate::LangChainError;

#[allow(async_fn_in_trait)]
pub trait BaseTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    async fn call(&self, input: &str) -> Result<String, LangChainError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDefinition {
    name: String,
    description: String,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> &str {
        &self.description
    }
}

pub fn tool(name: impl Into<String>, description: impl Into<String>) -> ToolDefinition {
    ToolDefinition::new(name, description)
}
