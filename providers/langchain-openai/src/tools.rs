use langchain_core::tools::{ToolDefinition, tool};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct CustomToolDefinition {
    definition: ToolDefinition,
    metadata: Value,
    format: Option<Value>,
}

impl CustomToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            definition: tool(name, description),
            metadata: serde_json::json!({ "type": "custom_tool" }),
            format: None,
        }
    }

    pub fn with_parameters(mut self, parameters: Value) -> Self {
        self.definition = self.definition.with_parameters(parameters);
        self
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.definition = self.definition.with_strict(strict);
        self
    }

    pub fn with_format(mut self, format: Value) -> Self {
        self.format = Some(format);
        self
    }

    pub fn name(&self) -> &str {
        self.definition.name()
    }

    pub fn description(&self) -> &str {
        self.definition.description()
    }

    pub fn parameters(&self) -> &Value {
        self.definition.parameters()
    }

    pub fn strict(&self) -> Option<bool> {
        self.definition.strict()
    }

    pub fn metadata(&self) -> &Value {
        &self.metadata
    }

    pub fn format(&self) -> Option<&Value> {
        self.format.as_ref()
    }

    pub fn as_tool_definition(&self) -> &ToolDefinition {
        &self.definition
    }

    pub fn into_tool_definition(self) -> ToolDefinition {
        self.definition
    }
}

pub fn custom_tool(
    name: impl Into<String>,
    description: impl Into<String>,
) -> CustomToolDefinition {
    CustomToolDefinition::new(name, description)
}
