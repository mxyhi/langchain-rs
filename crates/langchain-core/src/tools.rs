use std::sync::Arc;

use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::LangChainError;
use crate::messages::{ToolCall, ToolMessage, ToolMessageStatus};
use crate::runnables::RunnableConfig;

pub trait BaseTool: Send + Sync {
    fn definition(&self) -> &ToolDefinition;

    fn invoke<'a>(
        &'a self,
        input: ToolCall,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<ToolMessage, LangChainError>>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    name: String,
    description: String,
    parameters: Value,
    strict: Option<bool>,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: json!({
                "type": "object",
                "properties": {}
            }),
            strict: None,
        }
    }

    pub fn with_parameters(mut self, parameters: Value) -> Self {
        self.parameters = parameters;
        self
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn parameters(&self) -> &Value {
        &self.parameters
    }

    pub fn strict(&self) -> Option<bool> {
        self.strict
    }
}

pub fn tool(name: impl Into<String>, description: impl Into<String>) -> ToolDefinition {
    ToolDefinition::new(name, description)
}

type StructuredToolHandler =
    dyn Fn(Value) -> BoxFuture<'static, Result<Value, LangChainError>> + Send + Sync;

#[derive(Clone)]
pub struct StructuredTool {
    definition: ToolDefinition,
    handler: Arc<StructuredToolHandler>,
}

impl StructuredTool {
    pub fn new(
        definition: ToolDefinition,
        handler: impl Fn(Value) -> BoxFuture<'static, Result<Value, LangChainError>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            definition,
            handler: Arc::new(handler),
        }
    }
}

impl BaseTool for StructuredTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    fn invoke<'a>(
        &'a self,
        input: ToolCall,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<ToolMessage, LangChainError>> {
        Box::pin(async move {
            let output = (self.handler)(input.args().clone()).await?;
            let content = match &output {
                Value::String(content) => content.clone(),
                _ => output.to_string(),
            };

            Ok(ToolMessage::with_parts(
                content,
                input.id().unwrap_or_default(),
                Some(self.definition.name()),
                Some(output),
                ToolMessageStatus::Success,
            ))
        })
    }
}
