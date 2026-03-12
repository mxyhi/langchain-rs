use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::language_models::BaseLLM;
use langchain_core::outputs::{Generation, LLMResult};
use langchain_core::runnables::RunnableConfig;
use serde_json::{Value, json};

use crate::chat_models::{AnthropicContentBlock, AnthropicMessagesResponse, AnthropicUsage};
use crate::client::AnthropicClientConfig;

#[derive(Debug, Clone)]
pub struct AnthropicLLM {
    config: AnthropicClientConfig,
    model: String,
    max_tokens: usize,
    temperature: Option<f32>,
}

impl AnthropicLLM {
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
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens.max(1);
        self
    }

    fn request(&self, prompt: String) -> AnthropicLlmRequest {
        AnthropicLlmRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            messages: vec![AnthropicLlmMessage {
                role: "user".to_owned(),
                content: prompt,
            }],
        }
    }
}

impl BaseLLM for AnthropicLLM {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn generate<'a>(
        &'a self,
        prompts: Vec<String>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<LLMResult, LangChainError>> {
        Box::pin(async move {
            let mut generations = Vec::with_capacity(prompts.len());
            let mut total_usage = AnthropicUsage {
                input_tokens: 0,
                output_tokens: 0,
            };

            for prompt in prompts {
                let response = self
                    .config
                    .post("v1/messages")
                    .json(&self.request(prompt))
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

                let text = response
                    .content
                    .into_iter()
                    .filter_map(|block| match block {
                        AnthropicContentBlock::Text { text } => Some(text),
                        AnthropicContentBlock::ToolUse { .. } => None,
                    })
                    .collect::<String>();
                generations.push(vec![Generation::new(text)]);

                if let Some(usage) = response.usage {
                    total_usage.input_tokens += usage.input_tokens;
                    total_usage.output_tokens += usage.output_tokens;
                }
            }

            Ok(LLMResult::new(generations).with_output(BTreeMap::from([
                ("model_name".to_owned(), Value::String(self.model.clone())),
                (
                    "token_usage".to_owned(),
                    json!({
                        "prompt_tokens": total_usage.input_tokens,
                        "completion_tokens": total_usage.output_tokens,
                        "total_tokens": total_usage.input_tokens + total_usage.output_tokens,
                    }),
                ),
            ])))
        })
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct AnthropicLlmRequest {
    model: String,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    messages: Vec<AnthropicLlmMessage>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct AnthropicLlmMessage {
    role: String,
    content: String,
}
