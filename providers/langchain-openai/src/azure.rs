use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::embeddings::Embeddings;
use langchain_core::language_models::{BaseChatModel, BaseLLM};
use langchain_core::messages::{
    AIMessage, BaseMessage, InvalidToolCall, ResponseMetadata, ToolCall, UsageMetadata,
};
use langchain_core::outputs::{Generation, LLMResult};
use langchain_core::runnables::RunnableConfig;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const DEFAULT_API_VERSION: &str = "2024-02-01";

fn deployment_base_url(azure_endpoint: &str, deployment_name: &str) -> String {
    format!(
        "{}/openai/deployments/{}",
        azure_endpoint.trim_end_matches('/'),
        deployment_name
    )
}

fn azure_request(
    client: &Client,
    azure_endpoint: &str,
    deployment_name: &str,
    api_version: &str,
    api_key: Option<&str>,
    path: &str,
) -> reqwest::RequestBuilder {
    let url = format!(
        "{}/{}",
        deployment_base_url(azure_endpoint, deployment_name),
        path.trim_start_matches('/')
    );
    let mut request = client.post(url).query(&[("api-version", api_version)]);
    if let Some(api_key) = api_key {
        request = request.header("api-key", api_key);
    }
    request
}

#[derive(Debug, Clone)]
pub struct AzureChatOpenAI {
    client: Client,
    model_name: String,
    deployment_name: String,
    azure_endpoint: String,
    api_version: String,
    api_key: Option<String>,
}

impl AzureChatOpenAI {
    pub fn new(
        model_name: impl Into<String>,
        deployment_name: impl Into<String>,
        azure_endpoint: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            client: Client::new(),
            model_name: model_name.into(),
            deployment_name: deployment_name.into(),
            azure_endpoint: azure_endpoint.into(),
            api_version: DEFAULT_API_VERSION.to_owned(),
            api_key: api_key.map(|value| value.as_ref().to_owned()),
        }
    }

    pub fn deployment_name(&self) -> &str {
        &self.deployment_name
    }

    pub fn azure_endpoint(&self) -> &str {
        &self.azure_endpoint
    }

    pub fn api_version(&self) -> &str {
        &self.api_version
    }

    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    pub fn base_url(&self) -> String {
        deployment_base_url(&self.azure_endpoint, &self.deployment_name)
    }
}

impl BaseChatModel for AzureChatOpenAI {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn generate<'a>(
        &'a self,
        messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        Box::pin(async move {
            let response = azure_request(
                &self.client,
                &self.azure_endpoint,
                &self.deployment_name,
                &self.api_version,
                self.api_key.as_deref(),
                "chat/completions",
            )
            .json(&ChatCompletionsRequest {
                model: self.model_name.clone(),
                messages: messages.iter().map(OpenAIMessage::from_langchain).collect(),
            })
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
            let choice = response.choices.into_iter().next().ok_or_else(|| {
                LangChainError::request("azure openai response contained no choices")
            })?;

            let mut response_metadata = ResponseMetadata::new();
            response_metadata.insert("id".to_owned(), Value::String(response.id));
            response_metadata.insert("model".to_owned(), Value::String(response.model));

            let (tool_calls, invalid_tool_calls) = parse_tool_calls(choice.message.tool_calls);

            Ok(AIMessage::with_metadata(
                choice.message.content.unwrap_or_default(),
                response_metadata,
                response.usage.map(|usage| UsageMetadata {
                    input_tokens: usage.prompt_tokens,
                    output_tokens: usage.completion_tokens,
                    total_tokens: usage.total_tokens,
                }),
            )
            .with_tool_calls(tool_calls)
            .with_invalid_tool_calls(invalid_tool_calls))
        })
    }

    fn identifying_params(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            (
                "model_name".to_owned(),
                Value::String(self.model_name.clone()),
            ),
            (
                "deployment_name".to_owned(),
                Value::String(self.deployment_name.clone()),
            ),
            (
                "azure_endpoint".to_owned(),
                Value::String(self.azure_endpoint.clone()),
            ),
            (
                "api_version".to_owned(),
                Value::String(self.api_version.clone()),
            ),
        ])
    }
}

#[derive(Debug, Clone)]
pub struct AzureOpenAI {
    client: Client,
    model_name: String,
    deployment_name: String,
    azure_endpoint: String,
    api_version: String,
    api_key: Option<String>,
}

impl AzureOpenAI {
    pub fn new(
        model_name: impl Into<String>,
        deployment_name: impl Into<String>,
        azure_endpoint: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            client: Client::new(),
            model_name: model_name.into(),
            deployment_name: deployment_name.into(),
            azure_endpoint: azure_endpoint.into(),
            api_version: DEFAULT_API_VERSION.to_owned(),
            api_key: api_key.map(|value| value.as_ref().to_owned()),
        }
    }

    pub fn deployment_name(&self) -> &str {
        &self.deployment_name
    }

    pub fn azure_endpoint(&self) -> &str {
        &self.azure_endpoint
    }

    pub fn api_version(&self) -> &str {
        &self.api_version
    }

    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    pub fn base_url(&self) -> String {
        deployment_base_url(&self.azure_endpoint, &self.deployment_name)
    }
}

impl BaseLLM for AzureOpenAI {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn generate<'a>(
        &'a self,
        prompts: Vec<String>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<LLMResult, LangChainError>> {
        Box::pin(async move {
            let response = azure_request(
                &self.client,
                &self.azure_endpoint,
                &self.deployment_name,
                &self.api_version,
                self.api_key.as_deref(),
                "completions",
            )
            .json(&CompletionRequest {
                model: self.model_name.clone(),
                prompt: prompts.clone(),
            })
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
                .json::<CompletionResponse>()
                .await
                .map_err(|error| LangChainError::request(error.to_string()))?;

            let generations = vec![
                response
                    .choices
                    .into_iter()
                    .map(|choice| {
                        let mut info = BTreeMap::new();
                        info.insert(
                            "finish_reason".to_owned(),
                            choice
                                .finish_reason
                                .map(Value::String)
                                .unwrap_or(Value::Null),
                        );
                        Generation::with_info(choice.text, info)
                    })
                    .collect::<Vec<_>>(),
            ];

            Ok(LLMResult::new(generations).with_output(BTreeMap::from([
                (
                    "model_name".to_owned(),
                    Value::String(self.model_name.clone()),
                ),
                (
                    "token_usage".to_owned(),
                    json!({
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }),
                ),
            ])))
        })
    }

    fn identifying_params(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            (
                "model_name".to_owned(),
                Value::String(self.model_name.clone()),
            ),
            (
                "deployment_name".to_owned(),
                Value::String(self.deployment_name.clone()),
            ),
            (
                "azure_endpoint".to_owned(),
                Value::String(self.azure_endpoint.clone()),
            ),
            (
                "api_version".to_owned(),
                Value::String(self.api_version.clone()),
            ),
        ])
    }
}

#[derive(Debug, Clone)]
pub struct AzureOpenAIEmbeddings {
    client: Client,
    model_name: String,
    deployment_name: String,
    azure_endpoint: String,
    api_version: String,
    api_key: Option<String>,
}

impl AzureOpenAIEmbeddings {
    pub fn new(
        model_name: impl Into<String>,
        deployment_name: impl Into<String>,
        azure_endpoint: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            client: Client::new(),
            model_name: model_name.into(),
            deployment_name: deployment_name.into(),
            azure_endpoint: azure_endpoint.into(),
            api_version: DEFAULT_API_VERSION.to_owned(),
            api_key: api_key.map(|value| value.as_ref().to_owned()),
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn deployment_name(&self) -> &str {
        &self.deployment_name
    }

    pub fn azure_endpoint(&self) -> &str {
        &self.azure_endpoint
    }

    pub fn api_version(&self) -> &str {
        &self.api_version
    }

    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    pub fn base_url(&self) -> String {
        deployment_base_url(&self.azure_endpoint, &self.deployment_name)
    }

    pub fn identifying_params(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            (
                "model_name".to_owned(),
                Value::String(self.model_name.clone()),
            ),
            (
                "deployment_name".to_owned(),
                Value::String(self.deployment_name.clone()),
            ),
            (
                "azure_endpoint".to_owned(),
                Value::String(self.azure_endpoint.clone()),
            ),
            (
                "api_version".to_owned(),
                Value::String(self.api_version.clone()),
            ),
            ("boundary".to_owned(), json!("AzureOpenAIEmbeddings")),
        ])
    }
}

impl Embeddings for AzureOpenAIEmbeddings {
    fn embed_query<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>, LangChainError>> {
        Box::pin(async move {
            let mut embeddings = self.embed_documents(vec![text.to_owned()]).await?;
            embeddings.pop().ok_or_else(|| {
                LangChainError::request("azure embeddings response contained no vectors")
            })
        })
    }

    fn embed_documents<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, LangChainError>> {
        Box::pin(async move {
            let response = azure_request(
                &self.client,
                &self.azure_endpoint,
                &self.deployment_name,
                &self.api_version,
                self.api_key.as_deref(),
                "embeddings",
            )
            .json(&EmbeddingsRequest {
                model: self.model_name.clone(),
                input: texts,
            })
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
                .json::<EmbeddingsResponse>()
                .await
                .map_err(|error| LangChainError::request(error.to_string()))?;
            Ok(response
                .data
                .into_iter()
                .map(|item| item.embedding)
                .collect())
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct ChatCompletionsRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAIMessage {
    role: String,
    content: Option<String>,
}

impl OpenAIMessage {
    fn from_langchain(message: &BaseMessage) -> Self {
        Self {
            role: match message.role() {
                langchain_core::messages::MessageRole::System => "system",
                langchain_core::messages::MessageRole::Human => "user",
                langchain_core::messages::MessageRole::Ai => "assistant",
                langchain_core::messages::MessageRole::Tool => "tool",
            }
            .to_owned(),
            content: Some(message.content().to_owned()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionsResponse {
    id: String,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionChoice {
    message: ChatCompletionMessage,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OpenAIToolCall>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAIToolCall {
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAIToolFunction>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAIToolFunction {
    name: String,
    arguments: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct TokenUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
struct CompletionRequest {
    model: String,
    prompt: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct CompletionResponse {
    choices: Vec<CompletionChoice>,
    usage: TokenUsage,
}

#[derive(Debug, Clone, Deserialize)]
struct CompletionChoice {
    text: String,
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct EmbeddingsRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Clone, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

fn parse_tool_calls(raw_tool_calls: Vec<OpenAIToolCall>) -> (Vec<ToolCall>, Vec<InvalidToolCall>) {
    let mut tool_calls = Vec::new();
    let mut invalid_tool_calls = Vec::new();
    for raw_tool_call in raw_tool_calls {
        let Some(function) = raw_tool_call.function else {
            let invalid = InvalidToolCall::new(
                None::<String>,
                None::<String>,
                Some("missing function payload"),
            );
            invalid_tool_calls.push(match raw_tool_call.id {
                Some(id) => invalid.with_id(id),
                None => invalid,
            });
            continue;
        };
        match function.arguments {
            Value::Object(arguments) => {
                let mut tool_call = ToolCall::new(function.name, Value::Object(arguments));
                if let Some(id) = raw_tool_call.id {
                    tool_call = tool_call.with_id(id);
                }
                tool_calls.push(tool_call);
            }
            other => {
                let invalid = InvalidToolCall::new(
                    Some(function.name),
                    Some(other.to_string()),
                    Some("invalid tool call arguments"),
                );
                invalid_tool_calls.push(match raw_tool_call.id {
                    Some(id) => invalid.with_id(id),
                    None => invalid,
                });
            }
        }
    }
    (tool_calls, invalid_tool_calls)
}
