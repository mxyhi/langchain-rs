use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::language_models::BaseChatModel;
use langchain_core::messages::{AIMessage, BaseMessage, ResponseMetadata, UsageMetadata};
use langchain_core::runnables::RunnableConfig;
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct ChatOpenAI {
    client: Client,
    model: String,
    base_url: String,
    api_key: Option<SecretString>,
}

impl ChatOpenAI {
    pub fn new(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            client: Client::new(),
            model: model.into(),
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key: api_key.map(|key| SecretString::new(key.as_ref().to_owned().into())),
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url)
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
            let request = ChatCompletionsRequest {
                model: self.model.clone(),
                messages: messages.iter().map(OpenAIMessage::from_langchain).collect(),
            };

            let mut builder = self.client.post(self.endpoint()).json(&request);
            if let Some(api_key) = &self.api_key {
                builder = builder.bearer_auth(api_key.expose_secret());
            }

            let response = builder
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

            Ok(AIMessage::with_metadata(
                choice.message.content,
                response_metadata,
                usage_metadata,
            ))
        })
    }

    fn identifying_params(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([("model".to_owned(), Value::String(self.model.clone()))])
    }
}

#[derive(Debug, Clone, Serialize)]
struct ChatCompletionsRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

impl OpenAIMessage {
    fn from_langchain(message: &BaseMessage) -> Self {
        Self {
            role: message.role().as_openai_role().to_owned(),
            content: message.content().to_owned(),
        }
    }
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
