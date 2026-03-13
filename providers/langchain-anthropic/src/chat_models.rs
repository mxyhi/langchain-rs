use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::language_models::{BaseChatModel, ToolBindingOptions, ToolChoice};
use langchain_core::messages::{
    AIMessage, BaseMessage, MessageRole, ResponseMetadata, ToolCall, UsageMetadata,
};
use langchain_core::runnables::RunnableConfig;
use langchain_core::tools::ToolDefinition;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::client::AnthropicClientConfig;
use crate::middleware::CACHE_CONTROL_CONFIG_KEY;

#[derive(Debug, Clone)]
pub struct ChatAnthropic {
    config: AnthropicClientConfig,
    model: String,
    max_tokens: usize,
    temperature: Option<f32>,
    tools: Vec<ToolDefinition>,
    tool_choice: Option<AnthropicToolChoice>,
}

impl ChatAnthropic {
    pub fn new(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            config: AnthropicClientConfig::new(reqwest::Client::new(), base_url, api_key),
            model: model.into(),
            max_tokens: 1024,
            temperature: None,
            tools: Vec::new(),
            tool_choice: None,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens.max(1);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn base_url(&self) -> &str {
        self.config.base_url()
    }

    fn request(
        &self,
        messages: Vec<BaseMessage>,
        config: &RunnableConfig,
    ) -> Result<AnthropicMessagesRequest, LangChainError> {
        let mut system_parts = Vec::new();
        let mut body_messages = Vec::new();

        for message in messages {
            match message.role() {
                MessageRole::System => system_parts.push(message.content().to_owned()),
                MessageRole::Human => {
                    body_messages.push(AnthropicMessage::text("user", message.content()))
                }
                MessageRole::Ai => {
                    body_messages.push(AnthropicMessage::text("assistant", message.content()))
                }
                MessageRole::Tool => body_messages.push(AnthropicMessage::tool_result(message)),
            }
        }

        Ok(AnthropicMessagesRequest {
            model: self.model.clone(),
            system: (!system_parts.is_empty()).then(|| system_parts.join("\n\n")),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            messages: body_messages,
            cache_control: cache_control_from_config(config)?,
            tools: (!self.tools.is_empty()).then(|| {
                self.tools
                    .iter()
                    .map(convert_to_anthropic_tool)
                    .collect::<Vec<_>>()
            }),
            tool_choice: self.tool_choice.clone(),
        })
    }
}

impl BaseChatModel for ChatAnthropic {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn generate<'a>(
        &'a self,
        messages: Vec<BaseMessage>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        Box::pin(async move {
            let response = self
                .config
                .post("v1/messages")
                .json(&self.request(messages, &config)?)
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
                .json::<AnthropicMessagesResponse>()
                .await
                .map_err(|error| LangChainError::request(error.to_string()))?;

            let (content, tool_calls) = response.content.into_iter().fold(
                (String::new(), Vec::new()),
                |mut state, block| {
                    match block {
                        AnthropicContentBlock::Text { text } => state.0.push_str(&text),
                        AnthropicContentBlock::ToolUse { id, name, input } => {
                            state.1.push(ToolCall::new(name, input).with_id(id));
                        }
                    }
                    state
                },
            );

            let mut response_metadata = ResponseMetadata::new();
            response_metadata.insert("id".to_owned(), Value::String(response.id));
            response_metadata.insert("model".to_owned(), Value::String(response.model));
            response_metadata.insert(
                "stop_reason".to_owned(),
                response
                    .stop_reason
                    .map(Value::String)
                    .unwrap_or(Value::Null),
            );

            Ok(AIMessage::with_metadata(
                content,
                response_metadata,
                response.usage.map(|usage| UsageMetadata {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    total_tokens: usage.input_tokens + usage.output_tokens,
                }),
            )
            .with_tool_calls(tool_calls))
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
        let tool_choice = options.tool_choice.as_ref().map(AnthropicToolChoice::from);

        Ok(Box::new(Self {
            tools,
            tool_choice,
            ..self.clone()
        }))
    }
}

pub fn convert_to_anthropic_tool(tool: &ToolDefinition) -> AnthropicToolDefinition {
    AnthropicToolDefinition {
        name: tool.name().to_owned(),
        description: (!tool.description().is_empty()).then(|| tool.description().to_owned()),
        input_schema: tool.parameters().clone(),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnthropicToolDefinition {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    input_schema: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

impl From<&ToolChoice> for AnthropicToolChoice {
    fn from(value: &ToolChoice) -> Self {
        match value {
            ToolChoice::Auto => Self::Auto,
            ToolChoice::Any | ToolChoice::Required => Self::Any,
            ToolChoice::Named(name) => Self::Tool { name: name.clone() },
            ToolChoice::None => Self::Auto,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct AnthropicMessagesRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<AnthropicCacheControl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct AnthropicCacheControl {
    #[serde(rename = "type")]
    kind: String,
    ttl: String,
}

fn cache_control_from_config(
    config: &RunnableConfig,
) -> Result<Option<AnthropicCacheControl>, LangChainError> {
    config
        .metadata
        .get(CACHE_CONTROL_CONFIG_KEY)
        .cloned()
        .map(|value| {
            let cache_control: AnthropicCacheControl = serde_json::from_value(value).map_err(
                |error| {
                    LangChainError::request(format!(
                        "invalid Anthropic cache control metadata `{CACHE_CONTROL_CONFIG_KEY}`: {error}"
                    ))
                },
            )?;

            if cache_control.kind != "ephemeral" {
                return Err(LangChainError::request(format!(
                    "unsupported Anthropic cache control type `{}`; expected `ephemeral`",
                    cache_control.kind
                )));
            }

            if !matches!(cache_control.ttl.as_str(), "5m" | "1h") {
                return Err(LangChainError::request(format!(
                    "unsupported Anthropic cache control ttl `{}`; expected `5m` or `1h`",
                    cache_control.ttl
                )));
            }

            Ok(cache_control)
        })
        .transpose()
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicInputContentBlock>,
}

impl AnthropicMessage {
    fn text(role: &str, text: &str) -> Self {
        Self {
            role: role.to_owned(),
            content: vec![AnthropicInputContentBlock::Text {
                text: text.to_owned(),
            }],
        }
    }

    fn tool_result(message: BaseMessage) -> Self {
        let content = message.content().to_owned();
        let tool_use_id = match message {
            BaseMessage::Tool(message) => message.tool_call_id().to_owned(),
            _ => String::new(),
        };

        Self {
            role: "user".to_owned(),
            content: vec![AnthropicInputContentBlock::ToolResult {
                tool_use_id,
                content,
            }],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicInputContentBlock {
    Text {
        text: String,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AnthropicMessagesResponse {
    pub(crate) id: String,
    pub(crate) model: String,
    pub(crate) content: Vec<AnthropicContentBlock>,
    #[serde(default)]
    pub(crate) stop_reason: Option<String>,
    #[serde(default)]
    pub(crate) usage: Option<AnthropicUsage>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AnthropicUsage {
    pub(crate) input_tokens: usize,
    pub(crate) output_tokens: usize,
}
