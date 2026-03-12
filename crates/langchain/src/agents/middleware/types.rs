use std::sync::Arc;

use langchain_core::language_models::{BaseChatModel, ToolChoice};
use langchain_core::messages::{BaseMessage, SystemMessage};
use langchain_core::tools::ToolDefinition;
use serde_json::Value;

pub use crate::agents::AgentState;
use crate::agents::structured_output::ResponseFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JumpTo {
    Tools,
    Model,
    End,
}

#[derive(Clone)]
pub struct ModelRequest {
    model: Arc<dyn BaseChatModel>,
    messages: Vec<BaseMessage>,
    system_message: Option<SystemMessage>,
    tool_choice: Option<ToolChoice>,
    tools: Vec<ToolDefinition>,
    response_format: Option<ResponseFormat>,
    state: AgentState,
    jump_to: Option<JumpTo>,
}

impl ModelRequest {
    pub fn new(model: Arc<dyn BaseChatModel>, messages: Vec<BaseMessage>) -> Self {
        Self {
            model,
            messages,
            system_message: None,
            tool_choice: None,
            tools: Vec::new(),
            response_format: None,
            state: AgentState::new(Vec::new()),
            jump_to: None,
        }
    }

    pub fn with_system_message(mut self, system_message: SystemMessage) -> Self {
        self.system_message = Some(system_message);
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    pub fn with_state(mut self, state: AgentState) -> Self {
        self.state = state;
        self
    }

    pub fn with_jump_to(mut self, jump_to: JumpTo) -> Self {
        self.jump_to = Some(jump_to);
        self
    }

    pub fn override_with(&self) -> Self {
        self.clone()
    }

    pub fn model(&self) -> &Arc<dyn BaseChatModel> {
        &self.model
    }

    pub fn messages(&self) -> &[BaseMessage] {
        &self.messages
    }

    pub fn system_message(&self) -> Option<&SystemMessage> {
        self.system_message.as_ref()
    }

    pub fn tool_choice(&self) -> Option<&ToolChoice> {
        self.tool_choice.as_ref()
    }

    pub fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    pub fn response_format(&self) -> Option<&ResponseFormat> {
        self.response_format.as_ref()
    }

    pub fn state(&self) -> &AgentState {
        &self.state
    }

    pub fn jump_to(&self) -> Option<JumpTo> {
        self.jump_to
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelResponse {
    result: Vec<BaseMessage>,
    structured_response: Option<Value>,
}

impl ModelResponse {
    pub fn new(result: Vec<BaseMessage>) -> Self {
        Self {
            result,
            structured_response: None,
        }
    }

    pub fn with_structured_response(mut self, structured_response: Value) -> Self {
        self.structured_response = Some(structured_response);
        self
    }

    pub fn result(&self) -> &[BaseMessage] {
        &self.result
    }

    pub fn structured_response(&self) -> Option<&Value> {
        self.structured_response.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtendedModelResponse {
    model_response: ModelResponse,
    jump_to: Option<JumpTo>,
}

impl ExtendedModelResponse {
    pub fn new(model_response: ModelResponse) -> Self {
        Self {
            model_response,
            jump_to: None,
        }
    }

    pub fn with_jump_to(mut self, jump_to: JumpTo) -> Self {
        self.jump_to = Some(jump_to);
        self
    }

    pub fn model_response(&self) -> &ModelResponse {
        &self.model_response
    }

    pub fn jump_to(&self) -> Option<JumpTo> {
        self.jump_to
    }
}

pub trait AgentMiddleware {
    fn before_agent(&self, _state: &AgentState) -> Option<JumpTo> {
        None
    }

    fn after_model(&self, _response: &ModelResponse) -> Option<JumpTo> {
        None
    }
}
