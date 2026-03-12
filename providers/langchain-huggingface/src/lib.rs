use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::embeddings::Embeddings;
use langchain_core::language_models::{BaseChatModel, BaseLLM};
use langchain_core::messages::{AIMessage, BaseMessage};
use langchain_core::outputs::LLMResult;
use langchain_core::runnables::RunnableConfig;
use serde_json::Value;

fn unsupported_boundary(name: &str) -> LangChainError {
    LangChainError::unsupported(format!(
        "{name} is a real boundary type in this milestone, but its Hugging Face transport is not implemented yet"
    ))
}

#[derive(Debug, Clone)]
pub struct ChatHuggingFace {
    model_id: String,
}

impl ChatHuggingFace {
    pub fn from_model_id(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl BaseChatModel for ChatHuggingFace {
    fn model_name(&self) -> &str {
        &self.model_id
    }

    fn generate<'a>(
        &'a self,
        _messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        Box::pin(async move { Err(unsupported_boundary("ChatHuggingFace")) })
    }
}

#[derive(Debug, Clone)]
pub struct HuggingFaceEmbeddings {
    model_name: String,
}

impl HuggingFaceEmbeddings {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

impl Embeddings for HuggingFaceEmbeddings {
    fn embed_query<'a>(
        &'a self,
        _text: &'a str,
    ) -> BoxFuture<'a, Result<Vec<f32>, LangChainError>> {
        Box::pin(async move { Err(unsupported_boundary("HuggingFaceEmbeddings")) })
    }

    fn embed_documents<'a>(
        &'a self,
        _texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, LangChainError>> {
        Box::pin(async move { Err(unsupported_boundary("HuggingFaceEmbeddings")) })
    }
}

#[derive(Debug, Clone)]
pub struct HuggingFaceEndpointEmbeddings {
    inference_server_url: String,
}

impl HuggingFaceEndpointEmbeddings {
    pub fn new(inference_server_url: impl Into<String>) -> Self {
        Self {
            inference_server_url: inference_server_url.into(),
        }
    }

    pub fn inference_server_url(&self) -> &str {
        &self.inference_server_url
    }
}

impl Embeddings for HuggingFaceEndpointEmbeddings {
    fn embed_query<'a>(
        &'a self,
        _text: &'a str,
    ) -> BoxFuture<'a, Result<Vec<f32>, LangChainError>> {
        Box::pin(async move { Err(unsupported_boundary("HuggingFaceEndpointEmbeddings")) })
    }

    fn embed_documents<'a>(
        &'a self,
        _texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, LangChainError>> {
        Box::pin(async move { Err(unsupported_boundary("HuggingFaceEndpointEmbeddings")) })
    }
}

#[derive(Debug, Clone)]
pub struct HuggingFaceEndpoint {
    model_id: String,
    inference_server_url: Option<String>,
}

impl HuggingFaceEndpoint {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
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
}

impl BaseLLM for HuggingFaceEndpoint {
    fn model_name(&self) -> &str {
        &self.model_id
    }

    fn generate<'a>(
        &'a self,
        _prompts: Vec<String>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<LLMResult, LangChainError>> {
        Box::pin(async move { Err(unsupported_boundary("HuggingFaceEndpoint")) })
    }

    fn identifying_params(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([("model_id".to_owned(), Value::String(self.model_id.clone()))])
    }
}

#[derive(Debug, Clone)]
pub struct HuggingFacePipeline {
    model_id: String,
}

impl HuggingFacePipeline {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
        }
    }
}

impl BaseLLM for HuggingFacePipeline {
    fn model_name(&self) -> &str {
        &self.model_id
    }

    fn generate<'a>(
        &'a self,
        _prompts: Vec<String>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<LLMResult, LangChainError>> {
        Box::pin(async move { Err(unsupported_boundary("HuggingFacePipeline")) })
    }
}
