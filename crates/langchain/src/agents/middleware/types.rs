use std::collections::BTreeMap;
use std::sync::Arc;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::language_models::{BaseChatModel, ToolBindingOptions};
use langchain_core::messages::{BaseMessage, SystemMessage};
use langchain_core::tools::ToolDefinition;
use serde_json::Value;

pub use crate::agents::AgentState;
use crate::agents::structured_output::ResponseFormat;
pub use crate::tools::tool_node::{ToolCallHandler, ToolCallRequest, ToolCallWrapper};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JumpTo {
    Tools,
    Model,
    End,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HookConfig {
    can_jump_to: Vec<JumpTo>,
}

impl HookConfig {
    pub fn new(can_jump_to: Vec<JumpTo>) -> Self {
        Self { can_jump_to }
    }

    pub fn can_jump_to(&self) -> &[JumpTo] {
        &self.can_jump_to
    }
}

#[derive(Clone)]
pub struct ModelRequest {
    model: Arc<dyn BaseChatModel>,
    messages: Vec<BaseMessage>,
    system_message: Option<SystemMessage>,
    tool_binding_options: ToolBindingOptions,
    tools: Vec<ToolDefinition>,
    response_format: Option<ResponseFormat>,
    state: AgentState,
    model_settings: BTreeMap<String, Value>,
    jump_to: Option<JumpTo>,
}

impl ModelRequest {
    pub fn new(model: Arc<dyn BaseChatModel>, messages: Vec<BaseMessage>) -> Self {
        Self {
            model,
            messages,
            system_message: None,
            tool_binding_options: ToolBindingOptions::default(),
            tools: Vec::new(),
            response_format: None,
            state: AgentState::new(Vec::new()),
            model_settings: BTreeMap::new(),
            jump_to: None,
        }
    }

    pub fn with_system_message(mut self, system_message: SystemMessage) -> Self {
        self.system_message = Some(system_message);
        self
    }

    pub fn with_tool_choice(
        mut self,
        tool_choice: langchain_core::language_models::ToolChoice,
    ) -> Self {
        self.tool_binding_options.tool_choice = Some(tool_choice);
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

    pub fn with_tool_binding_options(mut self, options: ToolBindingOptions) -> Self {
        self.tool_binding_options = options;
        self
    }

    pub fn with_jump_to(mut self, jump_to: JumpTo) -> Self {
        self.jump_to = Some(jump_to);
        self
    }

    pub fn override_with(&self) -> Self {
        self.clone()
    }

    pub fn messages(&self) -> &[BaseMessage] {
        &self.messages
    }

    pub fn system_message(&self) -> Option<&SystemMessage> {
        self.system_message.as_ref()
    }

    pub fn tool_choice(&self) -> Option<&langchain_core::language_models::ToolChoice> {
        self.tool_binding_options.tool_choice.as_ref()
    }

    pub fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    pub fn response_format(&self) -> Option<&ResponseFormat> {
        self.response_format.as_ref()
    }

    pub fn messages_mut(&mut self) -> &mut Vec<BaseMessage> {
        &mut self.messages
    }

    pub fn tools_mut(&mut self) -> &mut Vec<ToolDefinition> {
        &mut self.tools
    }

    pub fn state_mut(&mut self) -> &mut AgentState {
        &mut self.state
    }

    pub fn state(&self) -> &AgentState {
        &self.state
    }

    pub fn tool_binding_options(&self) -> &ToolBindingOptions {
        &self.tool_binding_options
    }

    pub fn model(&self) -> &Arc<dyn BaseChatModel> {
        &self.model
    }

    pub fn with_model(mut self, model: Arc<dyn BaseChatModel>) -> Self {
        self.model = model;
        self
    }

    pub fn composed_messages(&self) -> Vec<BaseMessage> {
        match &self.system_message {
            Some(message) => {
                let mut messages = vec![BaseMessage::from(message.clone())];
                messages.extend(self.messages.clone());
                messages
            }
            None => self.messages.clone(),
        }
    }

    pub fn sync_state_messages(&mut self) {
        self.state = self.state.clone().with_messages(self.composed_messages());
    }

    pub fn model_settings(&self) -> &BTreeMap<String, Value> {
        &self.model_settings
    }

    pub fn model_settings_mut(&mut self) -> &mut BTreeMap<String, Value> {
        &mut self.model_settings
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

    pub fn result_mut(&mut self) -> &mut Vec<BaseMessage> {
        &mut self.result
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

pub type ModelCallResult = ModelResponse;
pub type ModelCallHandler = Arc<
    dyn Fn(ModelRequest) -> BoxFuture<'static, Result<ModelResponse, LangChainError>> + Send + Sync,
>;

pub trait AgentMiddleware: Send + Sync {
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    fn before_agent(&self, _state: &mut AgentState) -> Result<Option<JumpTo>, LangChainError> {
        Ok(None)
    }

    fn before_model(&self, _request: &mut ModelRequest) -> Result<Option<JumpTo>, LangChainError> {
        Ok(None)
    }

    fn after_model(
        &self,
        _state: &mut AgentState,
        _response: &mut ModelResponse,
    ) -> Result<Option<JumpTo>, LangChainError> {
        Ok(None)
    }

    fn after_agent(&self, _state: &mut AgentState) -> Result<Option<JumpTo>, LangChainError> {
        Ok(None)
    }

    fn wrap_model_call(
        &self,
        request: ModelRequest,
        handler: ModelCallHandler,
    ) -> BoxFuture<'static, Result<ModelResponse, LangChainError>> {
        handler(request)
    }

    fn wrap_tool_call(
        &self,
        request: ToolCallRequest,
        handler: ToolCallHandler,
    ) -> BoxFuture<'static, Result<langchain_core::messages::ToolMessage, LangChainError>> {
        handler(request)
    }
}

pub fn hook_config(can_jump_to: impl IntoIterator<Item = JumpTo>) -> HookConfig {
    HookConfig::new(can_jump_to.into_iter().collect())
}

pub fn before_agent<T>(middleware: T) -> T {
    middleware
}

pub fn before_model<T>(middleware: T) -> T {
    middleware
}

pub fn after_model<T>(middleware: T) -> T {
    middleware
}

pub fn after_agent<T>(middleware: T) -> T {
    middleware
}

pub fn dynamic_prompt(prompt: impl Into<String>) -> SystemMessage {
    SystemMessage::new(prompt)
}

pub fn wrap_model_call(
    request: ModelRequest,
    handler: ModelCallHandler,
) -> BoxFuture<'static, Result<ModelResponse, LangChainError>> {
    handler(request)
}

pub fn wrap_tool_call(
    request: ToolCallRequest,
    handler: ToolCallHandler,
) -> BoxFuture<'static, Result<langchain_core::messages::ToolMessage, LangChainError>> {
    handler(request)
}
