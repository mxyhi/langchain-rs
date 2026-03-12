use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::embeddings::Embeddings;
use langchain_core::language_models::{BaseChatModel, BaseLLM};
use langchain_core::messages::{AIMessage, BaseMessage};
use langchain_core::outputs::LLMResult;
use langchain_core::runnables::RunnableConfig;
use serde_json::{Value, json};

const DEFAULT_API_VERSION: &str = "2024-02-01";

fn unsupported_boundary(name: &str) -> LangChainError {
    LangChainError::unsupported(format!(
        "{name} transport is not implemented in this milestone"
    ))
}

fn deployment_base_url(azure_endpoint: &str, deployment_name: &str) -> String {
    format!(
        "{}/openai/deployments/{}",
        azure_endpoint.trim_end_matches('/'),
        deployment_name
    )
}

#[derive(Debug, Clone)]
pub struct AzureChatOpenAI {
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
        _messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        Box::pin(async move { Err(unsupported_boundary("AzureChatOpenAI")) })
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
        _prompts: Vec<String>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<LLMResult, LangChainError>> {
        Box::pin(async move { Err(unsupported_boundary("AzureOpenAI")) })
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
    fn embed_query<'a>(
        &'a self,
        _text: &'a str,
    ) -> BoxFuture<'a, Result<Vec<f32>, LangChainError>> {
        Box::pin(async move { Err(unsupported_boundary("AzureOpenAIEmbeddings")) })
    }

    fn embed_documents<'a>(
        &'a self,
        _texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, LangChainError>> {
        Box::pin(async move { Err(unsupported_boundary("AzureOpenAIEmbeddings")) })
    }
}
