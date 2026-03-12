use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::language_models::{
    BaseChatModel, StructuredOutput, StructuredOutputMethod, StructuredOutputOptions,
    StructuredOutputSchema, ToolBindingOptions, ToolChoice,
};
use langchain_core::messages::{
    AIMessage, BaseMessage, InvalidToolCall, ResponseMetadata, ToolCall, UsageMetadata,
};
use langchain_core::runnables::{Runnable, RunnableConfig, RunnableDyn};
use langchain_core::tools::ToolDefinition;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::client::OpenAIClientConfig;

#[derive(Debug, Clone)]
pub struct ChatOpenAI {
    config: OpenAIClientConfig,
    model: String,
    tools: Vec<ToolDefinition>,
    tool_choice: Option<ToolChoice>,
    parallel_tool_calls: Option<bool>,
    response_format: Option<ResponseFormat>,
}

impl ChatOpenAI {
    pub fn new(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            config: OpenAIClientConfig::new(reqwest::Client::new(), base_url, api_key),
            model: model.into(),
            tools: Vec::new(),
            tool_choice: None,
            parallel_tool_calls: None,
            response_format: None,
        }
    }

    pub fn bind_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    pub fn base_url(&self) -> &str {
        self.config.base_url()
    }

    pub fn with_tool_choice(mut self, tool_name: impl Into<String>) -> Self {
        self.tool_choice = Some(ToolChoice::Named(tool_name.into()));
        self
    }

    pub fn with_tool_choice_mode(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    pub fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = Some(parallel_tool_calls);
        self
    }

    fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    fn structured_output_runnable(
        &self,
        schema: ToolDefinition,
        method: StructuredOutputMethod,
        include_raw: bool,
    ) -> ProviderStructuredOutputRunnable {
        ProviderStructuredOutputRunnable {
            model: self.clone(),
            schema,
            method,
            include_raw,
        }
    }

    pub fn with_structured_output(
        &self,
        schema: ToolDefinition,
        method: StructuredOutputMethod,
        include_raw: bool,
    ) -> ProviderStructuredOutputRunnable {
        self.structured_output_runnable(schema, method, include_raw)
    }

    fn request(&self, messages: Vec<BaseMessage>) -> ChatCompletionsRequest {
        ChatCompletionsRequest {
            model: self.model.clone(),
            messages: messages.iter().map(OpenAIMessage::from_langchain).collect(),
            tools: self
                .tools
                .iter()
                .map(OpenAIToolDefinition::from_tool)
                .collect(),
            tool_choice: self.tool_choice.as_ref().map(tool_choice_to_wire),
            parallel_tool_calls: self.parallel_tool_calls,
            response_format: self.response_format.as_ref().map(ResponseFormat::to_wire),
        }
    }
}

impl BaseChatModel for ChatOpenAI {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn generate<'a>(
        &'a self,
        messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        Box::pin(async move {
            let response = self
                .config
                .post("chat/completions")
                .json(&self.request(messages))
                .send()
                .await
                .map_err(|error| LangChainError::request(error.to_string()))?;
            let status = response.status();

            if !status.is_success() {
                let body = response
                    .text()
                    .await
                    .unwrap_or_else(|_| String::from("<unreadable body>"));
                return Err(LangChainError::HttpStatus {
                    status: status.as_u16(),
                    body,
                });
            }

            let response = response
                .json::<ChatCompletionsResponse>()
                .await
                .map_err(|error| LangChainError::request(error.to_string()))?;

            let choice =
                response.choices.into_iter().next().ok_or_else(|| {
                    LangChainError::request("openai response contained no choices")
                })?;

            let mut response_metadata = ResponseMetadata::new();
            response_metadata.insert("id".to_owned(), Value::String(response.id));
            response_metadata.insert("model".to_owned(), Value::String(response.model));

            let usage_metadata = response.usage.map(|usage| UsageMetadata {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
            });

            let (tool_calls, invalid_tool_calls) = parse_tool_calls(choice.message.tool_calls);

            Ok(AIMessage::with_metadata(
                choice.message.content.unwrap_or_default(),
                response_metadata,
                usage_metadata,
            )
            .with_tool_calls(tool_calls)
            .with_invalid_tool_calls(invalid_tool_calls))
        })
    }

    fn identifying_params(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([("model".to_owned(), Value::String(self.model.clone()))])
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
        options: ToolBindingOptions,
    ) -> Result<Box<dyn BaseChatModel>, LangChainError> {
        let tools = tools
            .into_iter()
            .map(|tool| match (options.strict, tool.strict()) {
                (Some(strict), None) => tool.with_strict(strict),
                _ => tool,
            })
            .collect::<Vec<_>>();

        let mut model = self.clone().bind_tools(tools);
        if let Some(tool_choice) = options.tool_choice {
            model = model.with_tool_choice_mode(tool_choice);
        }
        if let Some(parallel_tool_calls) = options.parallel_tool_calls {
            model = model.with_parallel_tool_calls(parallel_tool_calls);
        }
        if let Some(response_format) = options.response_format {
            model = model.with_response_format(ResponseFormat::JsonSchema(
                response_format.to_tool_definition(options.strict),
            ));
        }

        Ok(Box::new(model))
    }

    fn with_structured_output(
        &self,
        schema: StructuredOutputSchema,
        options: StructuredOutputOptions,
    ) -> Result<Box<dyn RunnableDyn<Vec<BaseMessage>, StructuredOutput>>, LangChainError> {
        Ok(Box::new(self.structured_output_runnable(
            schema.to_tool_definition(options.strict),
            options.method,
            options.include_raw,
        )))
    }
}

#[derive(Debug, Clone)]
pub struct ProviderStructuredOutputRunnable {
    model: ChatOpenAI,
    schema: ToolDefinition,
    method: StructuredOutputMethod,
    include_raw: bool,
}

impl Runnable<Vec<BaseMessage>, StructuredOutput> for ProviderStructuredOutputRunnable {
    fn invoke<'a>(
        &'a self,
        input: Vec<BaseMessage>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<StructuredOutput, LangChainError>> {
        Box::pin(async move {
            let prepared_model = match self.method {
                StructuredOutputMethod::FunctionCalling => self
                    .model
                    .clone()
                    .bind_tools(vec![self.schema.clone()])
                    .with_tool_choice_mode(ToolChoice::Any)
                    .with_parallel_tool_calls(false),
                StructuredOutputMethod::JsonMode => self
                    .model
                    .clone()
                    .with_response_format(ResponseFormat::JsonObject),
                StructuredOutputMethod::JsonSchema => self
                    .model
                    .clone()
                    .with_response_format(ResponseFormat::JsonSchema(self.schema.clone())),
            };

            let raw = prepared_model.invoke(input, config).await?;
            let parsed = match self.method {
                StructuredOutputMethod::FunctionCalling => {
                    parse_structured_tool_output(&raw, self.schema.name())
                }
                StructuredOutputMethod::JsonMode | StructuredOutputMethod::JsonSchema => {
                    serde_json::from_str(raw.content()).map_err(LangChainError::from)
                }
            };

            match parsed {
                Ok(parsed) if self.include_raw => Ok(StructuredOutput::Raw {
                    raw,
                    parsed: Some(parsed),
                    parsing_error: None,
                }),
                Ok(parsed) => Ok(StructuredOutput::Parsed(parsed)),
                Err(error) if self.include_raw => Ok(StructuredOutput::Raw {
                    raw,
                    parsed: None,
                    parsing_error: Some(error.to_string()),
                }),
                Err(error) => Err(error),
            }
        })
    }
}

#[derive(Debug, Clone)]
enum ResponseFormat {
    JsonObject,
    JsonSchema(ToolDefinition),
}

impl ResponseFormat {
    fn to_wire(&self) -> Value {
        match self {
            Self::JsonObject => json!({ "type": "json_object" }),
            Self::JsonSchema(schema) => {
                let mut json_schema = json!({
                    "name": schema.name(),
                    "description": schema.description(),
                    "schema": schema.parameters(),
                });
                if let Some(strict) = schema.strict() {
                    json_schema["strict"] = Value::Bool(strict);
                }
                json!({
                    "type": "json_schema",
                    "json_schema": json_schema,
                })
            }
        }
    }
}

fn tool_choice_to_wire(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => Value::String("auto".to_owned()),
        ToolChoice::None => Value::String("none".to_owned()),
        ToolChoice::Required | ToolChoice::Any => Value::String("required".to_owned()),
        ToolChoice::Named(name) => json!({
            "type": "function",
            "function": { "name": name }
        }),
    }
}

#[derive(Debug, Clone, Serialize)]
struct ChatCompletionsRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAIToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

impl OpenAIMessage {
    fn from_langchain(message: &BaseMessage) -> Self {
        match message {
            BaseMessage::Human(message) => Self {
                role: "user".to_owned(),
                content: Some(message.content().to_owned()),
                tool_call_id: None,
                tool_calls: None,
            },
            BaseMessage::System(message) => Self {
                role: "system".to_owned(),
                content: Some(message.content().to_owned()),
                tool_call_id: None,
                tool_calls: None,
            },
            BaseMessage::Tool(message) => Self {
                role: "tool".to_owned(),
                content: Some(message.content().to_owned()),
                tool_call_id: Some(message.tool_call_id().to_owned()),
                tool_calls: None,
            },
            BaseMessage::Ai(message) => {
                let tool_calls = (!message.tool_calls().is_empty()
                    || !message.invalid_tool_calls().is_empty())
                .then(|| {
                    let mut serialized = message
                        .tool_calls()
                        .iter()
                        .map(OpenAIToolCall::from_tool_call)
                        .collect::<Vec<_>>();
                    serialized.extend(
                        message
                            .invalid_tool_calls()
                            .iter()
                            .map(OpenAIToolCall::from_invalid_tool_call),
                    );
                    serialized
                });

                let content = if tool_calls.is_some() && message.content().is_empty() {
                    None
                } else {
                    Some(message.content().to_owned())
                };

                Self {
                    role: "assistant".to_owned(),
                    content,
                    tool_call_id: None,
                    tool_calls,
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct OpenAIToolDefinition {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAIFunctionDefinition,
}

impl OpenAIToolDefinition {
    fn from_tool(tool: &ToolDefinition) -> Self {
        Self {
            kind: "function",
            function: OpenAIFunctionDefinition {
                name: tool.name().to_owned(),
                description: tool.description().to_owned(),
                parameters: tool.parameters().clone(),
                strict: tool.strict(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct OpenAIFunctionDefinition {
    name: String,
    description: String,
    parameters: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type")]
    kind: String,
    function: OpenAIFunctionCall,
}

impl OpenAIToolCall {
    fn from_tool_call(tool_call: &ToolCall) -> Self {
        Self {
            id: tool_call.id().map(str::to_owned),
            kind: "function".to_owned(),
            function: OpenAIFunctionCall {
                name: tool_call.name().to_owned(),
                arguments: tool_call.args().to_string(),
            },
        }
    }

    fn from_invalid_tool_call(tool_call: &InvalidToolCall) -> Self {
        Self {
            id: tool_call.id().map(str::to_owned),
            kind: "function".to_owned(),
            function: OpenAIFunctionCall {
                name: tool_call.name().unwrap_or_default().to_owned(),
                arguments: tool_call.raw_args().unwrap_or_default().to_owned(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionsResponse {
    id: String,
    model: String,
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize)]
struct Choice {
    message: OpenAIMessage,
}

#[derive(Debug, Clone, Deserialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

fn parse_tool_calls(
    raw_tool_calls: Option<Vec<OpenAIToolCall>>,
) -> (Vec<ToolCall>, Vec<InvalidToolCall>) {
    let mut parsed = Vec::new();
    let mut invalid = Vec::new();

    for raw_tool_call in raw_tool_calls.into_iter().flatten() {
        match serde_json::from_str::<Value>(&raw_tool_call.function.arguments) {
            Ok(args) => {
                let tool_call = ToolCall::new(raw_tool_call.function.name, args);
                parsed.push(match raw_tool_call.id {
                    Some(id) => tool_call.with_id(id),
                    None => tool_call,
                });
            }
            Err(error) => {
                let tool_call = InvalidToolCall::new(
                    Some(raw_tool_call.function.name),
                    Some(raw_tool_call.function.arguments),
                    Some(error.to_string()),
                );
                invalid.push(match raw_tool_call.id {
                    Some(id) => tool_call.with_id(id),
                    None => tool_call,
                });
            }
        }
    }

    (parsed, invalid)
}

fn parse_structured_tool_output(
    message: &AIMessage,
    tool_name: &str,
) -> Result<Value, LangChainError> {
    message
        .tool_calls()
        .iter()
        .find(|tool_call| tool_call.name() == tool_name)
        .map(|tool_call| tool_call.args().clone())
        .ok_or_else(|| {
            LangChainError::request(format!(
                "structured output response contained no tool call for `{tool_name}`"
            ))
        })
}
