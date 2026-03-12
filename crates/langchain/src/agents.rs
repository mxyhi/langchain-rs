use std::collections::BTreeMap;
use std::sync::Arc;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::language_models::{
    BaseChatModel, StructuredOutput, StructuredOutputOptions, StructuredOutputSchema,
    ToolBindingOptions, ToolChoice,
};
use langchain_core::messages::{AIMessage, BaseMessage, HumanMessage, SystemMessage};
use langchain_core::runnables::{Runnable, RunnableConfig, RunnableDyn};
use langchain_core::tools::ToolDefinition;
use serde_json::Value;

pub mod factory;
pub mod middleware;
pub mod structured_output;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentState {
    messages: Vec<BaseMessage>,
    structured_response: Option<Value>,
    metadata: BTreeMap<String, Value>,
}

impl AgentState {
    pub fn new(messages: Vec<BaseMessage>) -> Self {
        Self {
            messages,
            structured_response: None,
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_structured_response(mut self, structured_response: Value) -> Self {
        self.structured_response = Some(structured_response);
        self
    }

    pub fn with_messages(mut self, messages: Vec<BaseMessage>) -> Self {
        self.messages = messages;
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn messages(&self) -> &[BaseMessage] {
        &self.messages
    }

    pub fn structured_response(&self) -> Option<&Value> {
        self.structured_response.as_ref()
    }

    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }

    pub fn metadata_mut(&mut self) -> &mut BTreeMap<String, Value> {
        &mut self.metadata
    }
}

#[derive(Debug, Clone)]
pub struct MultipleStructuredOutputsError {
    tool_names: Vec<String>,
    ai_message: AIMessage,
}

impl MultipleStructuredOutputsError {
    pub fn new(tool_names: Vec<String>, ai_message: AIMessage) -> Self {
        Self {
            tool_names,
            ai_message,
        }
    }

    pub fn tool_names(&self) -> &[String] {
        &self.tool_names
    }

    pub fn ai_message(&self) -> &AIMessage {
        &self.ai_message
    }
}

impl std::fmt::Display for MultipleStructuredOutputsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Model incorrectly returned multiple structured responses ({}) when only one is expected.",
            self.tool_names.join(", ")
        )
    }
}

impl std::error::Error for MultipleStructuredOutputsError {}

impl MultipleStructuredOutputsError {
    fn into_langchain_error(self) -> LangChainError {
        LangChainError::request(self.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct StructuredOutputValidationError {
    tool_name: String,
    source: String,
    ai_message: AIMessage,
}

impl StructuredOutputValidationError {
    pub fn new(
        tool_name: impl Into<String>,
        source: impl Into<String>,
        ai_message: AIMessage,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            source: source.into(),
            ai_message,
        }
    }

    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn ai_message(&self) -> &AIMessage {
        &self.ai_message
    }
}

impl std::fmt::Display for StructuredOutputValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Failed to parse structured output for tool '{}': {}.",
            self.tool_name, self.source
        )
    }
}

impl std::error::Error for StructuredOutputValidationError {}

impl StructuredOutputValidationError {
    fn into_langchain_error(self) -> LangChainError {
        LangChainError::request(self.to_string())
    }
}

#[derive(Clone)]
pub struct Agent {
    model: Arc<dyn BaseChatModel>,
    system_prompt: Option<String>,
    tools: Vec<ToolDefinition>,
    tool_binding_options: ToolBindingOptions,
    response_format: Option<structured_output::ResponseFormat>,
    middlewares: Vec<Arc<dyn middleware::types::AgentMiddleware>>,
}

impl Agent {
    pub fn new(model: impl BaseChatModel + 'static) -> Self {
        Self {
            model: Arc::new(model),
            system_prompt: None,
            tools: Vec::new(),
            tool_binding_options: ToolBindingOptions::default(),
            response_format: None,
            middlewares: Vec::new(),
        }
    }

    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    pub fn with_tools(mut self, tools: Vec<ToolDefinition>, options: ToolBindingOptions) -> Self {
        self.tools = tools;
        self.tool_binding_options = options;
        self
    }

    pub fn with_response_format(mut self, response_format: StructuredOutputSchema) -> Self {
        self.response_format = Some(structured_output::ResponseFormat::Auto(
            structured_output::AutoStrategy::new(response_format),
        ));
        self
    }

    pub fn with_response_format_strategy(
        mut self,
        response_format: structured_output::ResponseFormat,
    ) -> Self {
        self.response_format = Some(response_format);
        self
    }

    pub fn with_middleware(
        mut self,
        middleware: impl middleware::types::AgentMiddleware + 'static,
    ) -> Self {
        self.middlewares.push(Arc::new(middleware));
        self
    }

    pub fn with_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn middleware::types::AgentMiddleware>>,
    ) -> Self {
        self.middlewares.extend(middlewares);
        self
    }

    fn validate_structured_output_message(
        &self,
        raw: &AIMessage,
        schema: &StructuredOutputSchema,
    ) -> Result<(), LangChainError> {
        let matching_tool_names = raw
            .tool_calls()
            .iter()
            .filter(|tool_call| tool_call.name() == schema.name())
            .map(|tool_call| tool_call.name().to_owned())
            .collect::<Vec<_>>();

        if matching_tool_names.len() > 1 {
            return Err(
                MultipleStructuredOutputsError::new(matching_tool_names, raw.clone())
                    .into_langchain_error(),
            );
        }

        Ok(())
    }

    async fn execute_model_request(
        &self,
        request: middleware::types::ModelRequest,
        config: RunnableConfig,
    ) -> Result<middleware::types::ModelResponse, LangChainError> {
        let messages = request.composed_messages();

        if let Some(response_format) = request.response_format() {
            let schema = response_format.schema().clone();
            let runnable: Box<dyn RunnableDyn<Vec<BaseMessage>, StructuredOutput>> =
                if request.tools().is_empty() {
                    request.model().with_structured_output(
                        schema.clone(),
                        StructuredOutputOptions {
                            include_raw: true,
                            ..StructuredOutputOptions::default()
                        },
                    )?
                } else {
                    request
                        .model()
                        .bind_tools(
                            request.tools().to_vec(),
                            ToolBindingOptions {
                                response_format: Some(schema.clone()),
                                tool_choice: Some(ToolChoice::Any),
                                parallel_tool_calls: Some(false),
                                ..request.tool_binding_options().clone()
                            },
                        )?
                        .with_structured_output(
                            schema.clone(),
                            StructuredOutputOptions {
                                include_raw: true,
                                ..StructuredOutputOptions::default()
                            },
                        )?
                };
            let output = runnable.invoke_boxed(messages.clone(), config).await?;

            let (raw, structured_response, parsing_error) = match output {
                StructuredOutput::Parsed(value) => (None, Some(value), None),
                StructuredOutput::Raw {
                    raw,
                    parsed,
                    parsing_error,
                } => (Some(raw), parsed, parsing_error),
            };

            if let Some(raw) = raw.as_ref() {
                self.validate_structured_output_message(raw, &schema)?;
            }

            if let Some(error) = parsing_error {
                let tool_name = raw
                    .as_ref()
                    .and_then(|message| {
                        message
                            .tool_calls()
                            .first()
                            .map(|tool_call| tool_call.name().to_owned())
                    })
                    .unwrap_or_else(|| schema.name().to_owned());
                let raw_message = raw.unwrap_or_else(|| AIMessage::new(""));

                return Err(
                    StructuredOutputValidationError::new(tool_name, error, raw_message)
                        .into_langchain_error(),
                );
            }

            let mut response = middleware::types::ModelResponse::new(Vec::new());
            if let Some(raw) = raw {
                response = middleware::types::ModelResponse::new(vec![BaseMessage::from(raw)]);
            }
            if let Some(structured_response) = structured_response {
                response = response.with_structured_response(structured_response);
            }
            return Ok(response);
        }

        let response = if request.tools().is_empty() {
            request.model().generate(messages, config).await?
        } else {
            request
                .model()
                .bind_tools(
                    request.tools().to_vec(),
                    request.tool_binding_options().clone(),
                )?
                .generate(messages, config)
                .await?
        };
        Ok(middleware::types::ModelResponse::new(vec![
            BaseMessage::from(response),
        ]))
    }

    pub async fn invoke_messages(
        &self,
        input: Vec<BaseMessage>,
        config: RunnableConfig,
    ) -> Result<AgentState, LangChainError> {
        let mut request = middleware::types::ModelRequest::new(self.model.clone(), input)
            .with_state(AgentState::new(Vec::new()))
            .with_tools(self.tools.clone())
            .with_tool_binding_options(self.tool_binding_options.clone());
        if let Some(system_prompt) = &self.system_prompt {
            request = request.with_system_message(SystemMessage::new(system_prompt.clone()));
        }
        if let Some(response_format) = &self.response_format {
            request = request.with_response_format(response_format.clone());
        }
        request.sync_state_messages();

        let mut state = request.state().clone();
        for middleware in &self.middlewares {
            if matches!(
                middleware.before_agent(&mut state)?,
                Some(middleware::types::JumpTo::End)
            ) {
                return Ok(state);
            }
        }
        request = request.with_state(state.clone());

        for middleware in &self.middlewares {
            if matches!(
                middleware.before_model(&mut request)?,
                Some(middleware::types::JumpTo::End)
            ) {
                let mut final_state = request.state().clone();
                final_state = final_state.with_messages(request.composed_messages());
                return Ok(final_state);
            }
        }
        request.sync_state_messages();
        state = request.state().clone();

        let config_clone = config.clone();
        let base_handler: middleware::types::ModelCallHandler = Arc::new({
            let agent = self.clone_for_handler();
            move |request| {
                let agent = agent.clone();
                let config = config_clone.clone();
                Box::pin(async move { agent.execute_model_request(request, config).await })
            }
        });
        let wrapped_handler = self
            .middlewares
            .iter()
            .rev()
            .fold(base_handler, |next, layer| {
                let middleware = layer.clone();
                Arc::new(move |request| middleware.wrap_model_call(request, next.clone()))
            });

        let mut response = wrapped_handler(request.clone()).await?;
        for middleware in &self.middlewares {
            if matches!(
                middleware.after_model(&mut state, &mut response)?,
                Some(middleware::types::JumpTo::End)
            ) {
                break;
            }
        }

        let mut messages = request.composed_messages();
        messages.extend(response.result().iter().cloned());
        let mut final_state = state.with_messages(messages);
        if let Some(structured_response) = response.structured_response() {
            final_state = final_state.with_structured_response(structured_response.clone());
        }

        for middleware in &self.middlewares {
            let _ = middleware.after_agent(&mut final_state)?;
        }

        Ok(final_state)
    }

    fn clone_for_handler(&self) -> Self {
        Self {
            model: self.model.clone(),
            system_prompt: self.system_prompt.clone(),
            tools: self.tools.clone(),
            tool_binding_options: self.tool_binding_options.clone(),
            response_format: self.response_format.clone(),
            middlewares: self.middlewares.clone(),
        }
    }
}

impl Runnable<String, AgentState> for Agent {
    fn invoke<'a>(
        &'a self,
        input: String,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AgentState, LangChainError>> {
        Box::pin(async move {
            self.invoke_messages(vec![BaseMessage::from(HumanMessage::new(input))], config)
                .await
        })
    }
}

pub fn create_agent(model: impl BaseChatModel + 'static) -> Agent {
    Agent::new(model)
}
