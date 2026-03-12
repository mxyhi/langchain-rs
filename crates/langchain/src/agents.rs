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

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentState {
    messages: Vec<BaseMessage>,
    structured_response: Option<Value>,
}

impl AgentState {
    pub fn new(messages: Vec<BaseMessage>) -> Self {
        Self {
            messages,
            structured_response: None,
        }
    }

    pub fn with_structured_response(mut self, structured_response: Value) -> Self {
        self.structured_response = Some(structured_response);
        self
    }

    pub fn messages(&self) -> &[BaseMessage] {
        &self.messages
    }

    pub fn structured_response(&self) -> Option<&Value> {
        self.structured_response.as_ref()
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

pub struct Agent {
    model: Box<dyn BaseChatModel>,
    system_prompt: Option<String>,
    tools: Vec<ToolDefinition>,
    tool_binding_options: ToolBindingOptions,
    response_format: Option<StructuredOutputSchema>,
}

impl Agent {
    pub fn new(model: impl BaseChatModel + 'static) -> Self {
        Self {
            model: Box::new(model),
            system_prompt: None,
            tools: Vec::new(),
            tool_binding_options: ToolBindingOptions::default(),
            response_format: None,
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
        self.response_format = Some(response_format);
        self
    }

    fn prepare_messages(&self, input: Vec<BaseMessage>) -> Vec<BaseMessage> {
        match &self.system_prompt {
            Some(system_prompt) => {
                let mut messages =
                    vec![BaseMessage::from(SystemMessage::new(system_prompt.clone()))];
                messages.extend(input);
                messages
            }
            None => input,
        }
    }

    pub async fn invoke_messages(
        &self,
        input: Vec<BaseMessage>,
        config: RunnableConfig,
    ) -> Result<AgentState, LangChainError> {
        let messages = self.prepare_messages(input);

        if let Some(schema) = &self.response_format {
            let runnable: Box<dyn RunnableDyn<Vec<BaseMessage>, StructuredOutput>> =
                if self.tools.is_empty() {
                    self.model.with_structured_output(
                        schema.clone(),
                        StructuredOutputOptions {
                            include_raw: true,
                            ..StructuredOutputOptions::default()
                        },
                    )?
                } else {
                    self.model
                        .bind_tools(
                            self.tools.clone(),
                            ToolBindingOptions {
                                response_format: Some(schema.clone()),
                                tool_choice: Some(ToolChoice::Any),
                                parallel_tool_calls: Some(false),
                                ..self.tool_binding_options.clone()
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

            if let Some(error) = parsing_error {
                return Err(LangChainError::request(error));
            }

            let mut state_messages = messages;
            if let Some(raw) = raw {
                state_messages.push(BaseMessage::from(raw));
            }

            let mut state = AgentState::new(state_messages);
            if let Some(structured_response) = structured_response {
                state = state.with_structured_response(structured_response);
            }
            return Ok(state);
        }

        let response = if self.tools.is_empty() {
            self.model.generate(messages.clone(), config).await?
        } else {
            self.model
                .bind_tools(self.tools.clone(), self.tool_binding_options.clone())?
                .generate(messages.clone(), config)
                .await?
        };
        let mut state_messages = messages;
        state_messages.push(BaseMessage::from(response));
        Ok(AgentState::new(state_messages))
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
