use std::collections::BTreeMap;

use langchain_core::LangChainError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::client::OpenAIClientConfig;

const DEFAULT_MODERATION_MODEL: &str = "omni-moderation-latest";

#[derive(Debug, Clone)]
pub struct OpenAIModerationClient {
    config: OpenAIClientConfig,
    model: String,
}

impl OpenAIModerationClient {
    pub fn new(base_url: impl Into<String>, api_key: Option<impl AsRef<str>>) -> Self {
        Self {
            config: OpenAIClientConfig::new(reqwest::Client::new(), base_url, api_key),
            model: DEFAULT_MODERATION_MODEL.to_owned(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub async fn moderate_text(
        &self,
        input: impl Into<String>,
    ) -> Result<OpenAIModerationResult, LangChainError> {
        let response = self
            .config
            .post("moderations")
            .json(&CreateModerationRequest {
                model: self.model.clone(),
                input: input.into(),
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
            .json::<CreateModerationResponse>()
            .await
            .map_err(|error| LangChainError::request(error.to_string()))?;

        response.results.into_iter().next().ok_or_else(|| {
            LangChainError::request("openai moderation response contained no results")
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct CreateModerationRequest {
    model: String,
    input: String,
}

#[derive(Debug, Clone, Deserialize)]
struct CreateModerationResponse {
    results: Vec<OpenAIModerationResult>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct OpenAIModerationResult {
    pub flagged: bool,
    #[serde(default)]
    pub categories: BTreeMap<String, Value>,
    #[serde(default)]
    pub category_scores: BTreeMap<String, Value>,
    #[serde(default)]
    pub category_applied_input_types: BTreeMap<String, Value>,
}
