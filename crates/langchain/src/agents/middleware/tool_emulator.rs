use serde_json::{Value, json};

use langchain_core::LangChainError;
use langchain_core::messages::ToolMessage;

use super::types::{AgentMiddleware, ToolCallRequest};

#[derive(Debug, Clone, PartialEq)]
pub struct EmulatedToolResult {
    content: String,
    artifact: Value,
}

impl EmulatedToolResult {
    pub fn new(content: impl Into<String>, artifact: Value) -> Self {
        Self {
            content: content.into(),
            artifact,
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn artifact(&self) -> &Value {
        &self.artifact
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LLMToolEmulator;

impl LLMToolEmulator {
    pub fn new() -> Self {
        Self
    }

    pub fn emulate(&self, request: &ToolCallRequest) -> Result<ToolMessage, LangChainError> {
        let result = self.emulate_result(request);
        Ok(ToolMessage::with_parts(
            result.content,
            request.tool_call().id().unwrap_or_default().to_owned(),
            Some(request.tool_call().name().to_owned()),
            Some(result.artifact),
            langchain_core::messages::ToolMessageStatus::Success,
        ))
    }

    pub fn emulate_result(&self, request: &ToolCallRequest) -> EmulatedToolResult {
        let tool_name = request.tool_call().name();
        let args = request.tool_call().args();
        let artifact = json!({
            "emulated": true,
            "tool_name": tool_name,
            "args": args,
            "state": request.state(),
        });

        let content = match args {
            Value::Object(map) if map.is_empty() => format!("Emulated tool `{tool_name}`"),
            _ => format!("Emulated tool `{tool_name}` with args {}", args),
        };

        EmulatedToolResult::new(content, artifact)
    }
}

impl AgentMiddleware for LLMToolEmulator {}
