use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::embeddings::Embeddings;
use langchain_core::language_models::{
    BaseChatModel, BaseLLM, StructuredOutput, StructuredOutputOptions, StructuredOutputSchema,
    ToolBindingOptions, ToolChoice,
};
use langchain_core::messages::{AIMessage, BaseMessage};
use langchain_core::outputs::LLMResult;
use langchain_core::runnables::{RunnableConfig, RunnableDyn};
use langchain_core::tools::ToolDefinition;
use langchain_openai::{
    OpenAICompatibleChatModel, OpenAICompatibleEmbeddings, OpenAICompatibleLlm,
};
use serde_json::Value;

const DEFAULT_BASE_URL: &str = "https://api.fireworks.ai/inference/v1";

#[derive(Debug, Clone)]
pub struct ChatFireworks {
    inner: OpenAICompatibleChatModel,
}

impl ChatFireworks {
    pub fn new(model: impl Into<String>, api_key: Option<impl AsRef<str>>) -> Self {
        Self::new_with_base_url(model, DEFAULT_BASE_URL, api_key)
    }

    pub fn new_with_base_url(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            inner: OpenAICompatibleChatModel::new(model, base_url, api_key),
        }
    }

    pub fn base_url(&self) -> &str {
        self.inner.base_url()
    }

    pub fn bind_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.inner = self.inner.bind_tools(tools);
        self
    }

    pub fn with_tool_choice(mut self, tool_name: impl Into<String>) -> Self {
        self.inner = self.inner.with_tool_choice(tool_name);
        self
    }

    pub fn with_tool_choice_mode(mut self, choice: ToolChoice) -> Self {
        self.inner = self.inner.with_tool_choice_mode(choice);
        self
    }

    pub fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.inner = self.inner.with_parallel_tool_calls(parallel_tool_calls);
        self
    }
}

impl BaseChatModel for ChatFireworks {
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn generate<'a>(
        &'a self,
        messages: Vec<BaseMessage>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        self.inner.generate(messages, config)
    }

    fn identifying_params(&self) -> BTreeMap<String, Value> {
        self.inner.identifying_params()
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
        options: ToolBindingOptions,
    ) -> Result<Box<dyn BaseChatModel>, LangChainError> {
        BaseChatModel::bind_tools(&self.inner, tools, options)
    }

    fn with_structured_output(
        &self,
        schema: StructuredOutputSchema,
        options: StructuredOutputOptions,
    ) -> Result<Box<dyn RunnableDyn<Vec<BaseMessage>, StructuredOutput>>, LangChainError> {
        BaseChatModel::with_structured_output(&self.inner, schema, options)
    }
}

#[derive(Debug, Clone)]
pub struct Fireworks {
    inner: OpenAICompatibleLlm,
}

impl Fireworks {
    pub fn new(model: impl Into<String>, api_key: Option<impl AsRef<str>>) -> Self {
        Self::new_with_base_url(model, DEFAULT_BASE_URL, api_key)
    }

    pub fn new_with_base_url(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            inner: OpenAICompatibleLlm::new(model, base_url, api_key),
        }
    }

    pub fn base_url(&self) -> &str {
        self.inner.base_url()
    }
}

impl BaseLLM for Fireworks {
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn generate<'a>(
        &'a self,
        prompts: Vec<String>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<LLMResult, LangChainError>> {
        self.inner.generate(prompts, config)
    }

    fn identifying_params(&self) -> BTreeMap<String, Value> {
        self.inner.identifying_params()
    }
}

#[derive(Debug, Clone)]
pub struct FireworksEmbeddings {
    inner: OpenAICompatibleEmbeddings,
}

impl FireworksEmbeddings {
    pub fn new(model: impl Into<String>, api_key: Option<impl AsRef<str>>) -> Self {
        Self::new_with_base_url(model, DEFAULT_BASE_URL, api_key)
    }

    pub fn new_with_base_url(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            inner: OpenAICompatibleEmbeddings::new(model, base_url, api_key),
        }
    }

    pub fn base_url(&self) -> &str {
        self.inner.base_url()
    }
}

impl Embeddings for FireworksEmbeddings {
    fn embed_query<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>, LangChainError>> {
        self.inner.embed_query(text)
    }

    fn embed_documents<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, LangChainError>> {
        self.inner.embed_documents(texts)
    }
}
