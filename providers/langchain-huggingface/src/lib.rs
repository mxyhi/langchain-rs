use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::embeddings::Embeddings;
use langchain_core::language_models::{BaseChatModel, BaseLLM};
use langchain_core::messages::{
    AIMessage, BaseMessage, MessageRole, ResponseMetadata, UsageMetadata,
};
use langchain_core::outputs::{Generation, LLMResult};
use langchain_core::runnables::RunnableConfig;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const DEFAULT_CHAT_BASE_URL: &str = "https://router.huggingface.co/v1";
const DEFAULT_INFERENCE_BASE_URL: &str = "https://router.huggingface.co/hf-inference/models";

pub mod data;

fn default_hf_api_key() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGINGFACEHUB_API_TOKEN").ok())
}

fn trim_base_url(base_url: impl Into<String>) -> String {
    base_url.into().trim_end_matches('/').to_owned()
}

fn default_model_endpoint(model_name: &str) -> String {
    format!("{DEFAULT_INFERENCE_BASE_URL}/{model_name}")
}

fn hf_router_model_name(model_id: &str) -> String {
    if model_id.contains(':') {
        model_id.to_owned()
    } else {
        format!("{model_id}:hf-inference")
    }
}

fn feature_extraction_endpoint(base_url: impl Into<String>, model_name: &str) -> String {
    let base_url = trim_base_url(base_url);
    if base_url.contains("/hf-inference/models/") || base_url.ends_with(model_name) {
        base_url
    } else {
        format!("{base_url}/hf-inference/models/{model_name}")
    }
}

#[derive(Debug, Clone)]
pub struct ChatHuggingFace {
    client: Client,
    model_id: String,
    base_url: String,
    api_key: Option<String>,
}

impl ChatHuggingFace {
    pub fn from_model_id(model_id: impl Into<String>) -> Self {
        Self::new_with_base_url(model_id, DEFAULT_CHAT_BASE_URL, default_hf_api_key())
    }

    pub fn new_with_base_url(
        model_id: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            client: Client::new(),
            model_id: model_id.into(),
            base_url: trim_base_url(base_url),
            api_key: api_key.map(|value| value.as_ref().to_owned()),
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

impl BaseChatModel for ChatHuggingFace {
    fn model_name(&self) -> &str {
        &self.model_id
    }

    fn generate<'a>(
        &'a self,
        messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        Box::pin(async move {
            let mut request = self
                .client
                .post(format!("{}/chat/completions", self.base_url))
                .json(&ChatCompletionsRequest {
                    model: hf_router_model_name(&self.model_id),
                    messages: messages
                        .into_iter()
                        .map(HfMessage::from_langchain)
                        .collect(),
                });
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
                LangChainError::request("huggingface response contained no choices")
            })?;

            let mut response_metadata = ResponseMetadata::new();
            response_metadata.insert("id".to_owned(), Value::String(response.id));
            response_metadata.insert("model".to_owned(), Value::String(response.model));

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
}

#[derive(Debug, Clone)]
pub struct HuggingFaceEmbeddings {
    client: Client,
    model_name: String,
    base_url: String,
    api_key: Option<String>,
}

impl HuggingFaceEmbeddings {
    pub fn new(model_name: impl Into<String>) -> Self {
        let model_name = model_name.into();
        Self::new_with_base_url(
            &model_name,
            default_model_endpoint(&model_name),
            default_hf_api_key(),
        )
    }

    pub fn new_with_base_url(
        model_name: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        let model_name = model_name.into();
        Self {
            client: Client::new(),
            base_url: feature_extraction_endpoint(base_url, &model_name),
            model_name,
            api_key: api_key.map(|value| value.as_ref().to_owned()),
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

impl Embeddings for HuggingFaceEmbeddings {
    fn embed_query<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>, LangChainError>> {
        Box::pin(async move {
            let mut embeddings = embed_huggingface(
                &self.client,
                &self.base_url,
                self.api_key.as_deref(),
                vec![text.to_owned()],
            )
            .await?;
            embeddings.pop().ok_or_else(|| {
                LangChainError::request("huggingface embeddings response contained no vectors")
            })
        })
    }

    fn embed_documents<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, LangChainError>> {
        Box::pin(async move {
            embed_huggingface(&self.client, &self.base_url, self.api_key.as_deref(), texts).await
        })
    }
}

#[derive(Debug, Clone)]
pub struct HuggingFaceEndpointEmbeddings {
    client: Client,
    inference_server_url: String,
    api_key: Option<String>,
}

impl HuggingFaceEndpointEmbeddings {
    pub fn new(inference_server_url: impl Into<String>) -> Self {
        Self::new_with_base_url(inference_server_url, default_hf_api_key())
    }

    pub fn new_with_base_url(
        inference_server_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            client: Client::new(),
            inference_server_url: trim_base_url(inference_server_url),
            api_key: api_key.map(|value| value.as_ref().to_owned()),
        }
    }

    pub fn inference_server_url(&self) -> &str {
        &self.inference_server_url
    }
}

impl Embeddings for HuggingFaceEndpointEmbeddings {
    fn embed_query<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>, LangChainError>> {
        Box::pin(async move {
            let mut embeddings = embed_huggingface(
                &self.client,
                &self.inference_server_url,
                self.api_key.as_deref(),
                vec![text.to_owned()],
            )
            .await?;
            embeddings.pop().ok_or_else(|| {
                LangChainError::request(
                    "huggingface endpoint embeddings response contained no vectors",
                )
            })
        })
    }

    fn embed_documents<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, LangChainError>> {
        Box::pin(async move {
            embed_huggingface(
                &self.client,
                &self.inference_server_url,
                self.api_key.as_deref(),
                texts,
            )
            .await
        })
    }
}

#[derive(Debug, Clone)]
pub struct HuggingFaceEndpoint {
    client: Client,
    model_id: String,
    base_url: String,
    api_key: Option<String>,
    inference_server_url: Option<String>,
}

impl HuggingFaceEndpoint {
    pub fn new(model_id: impl Into<String>) -> Self {
        let model_id = model_id.into();
        Self::new_with_base_url(
            &model_id,
            default_model_endpoint(&model_id),
            default_hf_api_key(),
        )
    }

    pub fn new_with_base_url(
        model_id: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            client: Client::new(),
            model_id: model_id.into(),
            base_url: trim_base_url(base_url),
            api_key: api_key.map(|value| value.as_ref().to_owned()),
            inference_server_url: None,
        }
    }

    pub fn with_inference_server_url(mut self, inference_server_url: impl Into<String>) -> Self {
        self.inference_server_url = Some(inference_server_url.into());
        self
    }

    pub fn inference_server_url(&self) -> Option<&str> {
        self.inference_server_url.as_deref()
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

impl BaseLLM for HuggingFaceEndpoint {
    fn model_name(&self) -> &str {
        &self.model_id
    }

    fn generate<'a>(
        &'a self,
        prompts: Vec<String>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<LLMResult, LangChainError>> {
        Box::pin(async move {
            if prompts.is_empty() {
                return Ok(LLMResult::new(Vec::<Vec<Generation>>::new()));
            }

            let mut generations = Vec::with_capacity(prompts.len());
            for prompt in prompts {
                let generated_text = generate_text(
                    &self.client,
                    &self.base_url,
                    self.api_key.as_deref(),
                    &prompt,
                )
                .await?;
                generations.push(vec![Generation::new(generated_text)]);
            }

            Ok(LLMResult::new(generations).with_output(BTreeMap::from([(
                "model_name".to_owned(),
                Value::String(self.model_id.clone()),
            )])))
        })
    }

    fn identifying_params(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("model_id".to_owned(), Value::String(self.model_id.clone())),
            ("base_url".to_owned(), Value::String(self.base_url.clone())),
        ])
    }
}

#[derive(Debug, Clone)]
pub struct HuggingFacePipeline {
    model_id: String,
}

impl HuggingFacePipeline {
    pub const UNAVAILABILITY_REASON: &str = "HuggingFacePipeline is a boundary marker for local transformers pipelines and is not exposed as a runnable Rust BaseLLM";

    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_id
    }

    pub const fn is_available(&self) -> bool {
        false
    }

    pub const fn unavailability_reason(&self) -> &'static str {
        Self::UNAVAILABILITY_REASON
    }
}

async fn embed_huggingface(
    client: &Client,
    base_url: &str,
    api_key: Option<&str>,
    texts: Vec<String>,
) -> Result<Vec<Vec<f32>>, LangChainError> {
    let mut request = client
        .post(base_url)
        .json(&FeatureExtractionRequest { inputs: texts });
    if let Some(api_key) = api_key {
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
        .json::<FeatureExtractionResponse>()
        .await
        .map_err(|error| LangChainError::request(error.to_string()))?;

    Ok(match response {
        FeatureExtractionResponse::Single(vector) => vec![vector],
        FeatureExtractionResponse::Batch(vectors) => vectors,
    })
}

async fn generate_text(
    client: &Client,
    base_url: &str,
    api_key: Option<&str>,
    prompt: &str,
) -> Result<String, LangChainError> {
    let mut request = client.post(base_url).json(&TextGenerationRequest {
        inputs: prompt.to_owned(),
    });
    if let Some(api_key) = api_key {
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
        .json::<TextGenerationResponse>()
        .await
        .map_err(|error| LangChainError::request(error.to_string()))?;

    match response {
        TextGenerationResponse::Single(item) => Ok(item.generated_text),
        TextGenerationResponse::Batch(mut items) => items
            .drain(..)
            .next()
            .map(|item| item.generated_text)
            .ok_or_else(|| LangChainError::request("huggingface endpoint returned no generations")),
    }
}

#[derive(Debug, Clone, Serialize)]
struct ChatCompletionsRequest {
    model: String,
    messages: Vec<HfMessage>,
}

#[derive(Debug, Clone, Serialize)]
struct HfMessage {
    role: &'static str,
    content: String,
}

impl HfMessage {
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

#[derive(Debug, Clone, Serialize)]
struct FeatureExtractionRequest {
    inputs: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum FeatureExtractionResponse {
    Single(Vec<f32>),
    Batch(Vec<Vec<f32>>),
}

#[derive(Debug, Clone, Serialize)]
struct TextGenerationRequest {
    inputs: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum TextGenerationResponse {
    Single(TextGenerationItem),
    Batch(Vec<TextGenerationItem>),
}

#[derive(Debug, Clone, Deserialize)]
struct TextGenerationItem {
    generated_text: String,
}

pub mod chat_models {
    pub use crate::ChatHuggingFace;
}

pub mod embeddings {
    pub use crate::{HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings};
}

pub mod llms {
    pub use crate::HuggingFaceEndpoint;
}

pub mod pipelines {
    pub use crate::HuggingFacePipeline;
}
