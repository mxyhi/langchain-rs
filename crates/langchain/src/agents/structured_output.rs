use langchain_core::language_models::StructuredOutputSchema;

pub use crate::agents::{MultipleStructuredOutputsError, StructuredOutputValidationError};

#[derive(Debug, Clone, PartialEq)]
pub struct AutoStrategy {
    schema: StructuredOutputSchema,
}

impl AutoStrategy {
    pub fn new(schema: StructuredOutputSchema) -> Self {
        Self { schema }
    }

    pub fn schema(&self) -> &StructuredOutputSchema {
        &self.schema
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderStrategy {
    schema: StructuredOutputSchema,
    strict: Option<bool>,
}

impl ProviderStrategy {
    pub fn new(schema: StructuredOutputSchema) -> Self {
        Self {
            schema,
            strict: None,
        }
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    pub fn schema(&self) -> &StructuredOutputSchema {
        &self.schema
    }

    pub fn strict(&self) -> Option<bool> {
        self.strict
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolStrategy {
    schema: StructuredOutputSchema,
    tool_message_content: Option<String>,
}

impl ToolStrategy {
    pub fn new(schema: StructuredOutputSchema) -> Self {
        Self {
            schema,
            tool_message_content: None,
        }
    }

    pub fn with_tool_message_content(mut self, content: impl Into<String>) -> Self {
        self.tool_message_content = Some(content.into());
        self
    }

    pub fn schema(&self) -> &StructuredOutputSchema {
        &self.schema
    }

    pub fn tool_message_content(&self) -> Option<&str> {
        self.tool_message_content.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResponseFormat {
    Auto(AutoStrategy),
    Provider(ProviderStrategy),
    Tool(ToolStrategy),
}

impl ResponseFormat {
    pub fn schema(&self) -> &StructuredOutputSchema {
        match self {
            Self::Auto(strategy) => strategy.schema(),
            Self::Provider(strategy) => strategy.schema(),
            Self::Tool(strategy) => strategy.schema(),
        }
    }
}
