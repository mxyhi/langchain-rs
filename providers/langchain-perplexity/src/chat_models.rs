use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::language_models::BaseChatModel;
use langchain_core::messages::{
    AIMessage, BaseMessage, MessageRole, ResponseMetadata, UsageMetadata,
};
use langchain_core::runnables::RunnableConfig;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::WebSearchOptions;

const DEFAULT_BASE_URL: &str = "https://api.perplexity.ai";

#[derive(Debug, Clone)]
pub struct ChatPerplexity {
    client: Client,
    model: String,
    base_url: String,
    api_key: Option<String>,
    web_search_options: Option<WebSearchOptions>,
}

impl ChatPerplexity {
    pub fn new(model: impl Into<String>) -> Self {
        Self::new_with_base_url(
            model,
            DEFAULT_BASE_URL,
            std::env::var("PERPLEXITY_API_KEY")
                .ok()
                .or_else(|| std::env::var("PPLX_API_KEY").ok()),
        )
    }

    pub fn new_with_base_url(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            client: Client::new(),
            model: model.into(),
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key: api_key.map(|value| value.as_ref().to_owned()),
            web_search_options: None,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn with_web_search_options(mut self, web_search_options: WebSearchOptions) -> Self {
        self.web_search_options = Some(web_search_options);
        self
    }

    fn request(&self, messages: Vec<BaseMessage>) -> ChatCompletionsRequest {
        ChatCompletionsRequest {
            model: self.model.clone(),
            messages: messages
                .into_iter()
                .map(PerplexityMessage::from_langchain)
                .collect(),
            web_search_options: self.web_search_options.clone(),
        }
    }
}

impl BaseChatModel for ChatPerplexity {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn generate<'a>(
        &'a self,
        messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        Box::pin(async move {
            let mut request = self
                .client
                .post(format!("{}/v1/sonar", self.base_url))
                .json(&self.request(messages));

            if let Some(api_key) = &self.api_key {
                request = request.bearer_auth(api_key);
            }

            let response = request
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
                LangChainError::request("perplexity response contained no choices")
            })?;

            let mut response_metadata = ResponseMetadata::new();
            response_metadata.insert("id".to_owned(), Value::String(response.id));
            response_metadata.insert("model".to_owned(), Value::String(response.model));
            if let Some(citations) = response.citations {
                response_metadata.insert(
                    "citations".to_owned(),
                    serde_json::to_value(citations).map_err(LangChainError::from)?,
                );
            }

            Ok(AIMessage::with_metadata(
                choice.message.content.unwrap_or_default(),
                response_metadata,
                response.usage.map(|usage| UsageMetadata {
                    input_tokens: usage.prompt_tokens,
                    output_tokens: usage.completion_tokens,
                    total_tokens: usage.total_tokens,
                }),
            ))
        })
    }

    fn identifying_params(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("model".to_owned(), Value::String(self.model.clone())),
            ("base_url".to_owned(), Value::String(self.base_url.clone())),
        ])
    }
}

#[derive(Debug, Clone, Serialize)]
struct ChatCompletionsRequest {
    model: String,
    messages: Vec<PerplexityMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    web_search_options: Option<WebSearchOptions>,
}

#[derive(Debug, Clone, Serialize)]
struct PerplexityMessage {
    role: &'static str,
    content: String,
}

impl PerplexityMessage {
    fn from_langchain(message: BaseMessage) -> Self {
        let role = match message.role() {
            MessageRole::System => "system",
            MessageRole::Human => "user",
            MessageRole::Ai => "assistant",
            MessageRole::Tool => "tool",
        };

        Self {
            role,
            content: message.content().to_owned(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionsResponse {
    id: String,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: Option<TokenUsage>,
    citations: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionChoice {
    message: ChatCompletionMessage,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionMessage {
    content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TokenUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}
