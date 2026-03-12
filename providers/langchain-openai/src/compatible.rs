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
use serde_json::Value;

use crate::{ChatOpenAI, OpenAI, OpenAIEmbeddings};

#[derive(Debug, Clone)]
pub struct OpenAICompatibleChatModel {
    inner: ChatOpenAI,
}

impl OpenAICompatibleChatModel {
    pub fn new(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            inner: ChatOpenAI::new(model, base_url, api_key),
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

impl BaseChatModel for OpenAICompatibleChatModel {
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
pub struct OpenAICompatibleLlm {
    inner: OpenAI,
}

impl OpenAICompatibleLlm {
    pub fn new(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            inner: OpenAI::new(model, base_url, api_key),
        }
    }

    pub fn base_url(&self) -> &str {
        self.inner.base_url()
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.inner = self.inner.with_temperature(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.inner = self.inner.with_max_tokens(max_tokens);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.inner = self.inner.with_top_p(top_p);
        self
    }

    pub fn with_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.inner = self.inner.with_frequency_penalty(frequency_penalty);
        self
    }

    pub fn with_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.inner = self.inner.with_presence_penalty(presence_penalty);
        self
    }

    pub fn with_n(mut self, n: usize) -> Self {
        self.inner = self.inner.with_n(n);
        self
    }

    pub fn with_best_of(mut self, best_of: usize) -> Self {
        self.inner = self.inner.with_best_of(best_of);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.inner = self.inner.with_seed(seed);
        self
    }

    pub fn with_logprobs(mut self, logprobs: u8) -> Self {
        self.inner = self.inner.with_logprobs(logprobs);
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.inner = self.inner.with_batch_size(batch_size);
        self
    }
}

impl BaseLLM for OpenAICompatibleLlm {
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
pub struct OpenAICompatibleEmbeddings {
    inner: OpenAIEmbeddings,
}

impl OpenAICompatibleEmbeddings {
    pub fn new(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            inner: OpenAIEmbeddings::new(model, base_url, api_key),
        }
    }

    pub fn base_url(&self) -> &str {
        self.inner.base_url()
    }

    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.inner = self.inner.with_dimensions(dimensions);
        self
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.inner = self.inner.with_chunk_size(chunk_size);
        self
    }
}

impl Embeddings for OpenAICompatibleEmbeddings {
    fn embed_query<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>, LangChainError>> {
        Box::pin(async move { self.inner.embed_query(text).await })
    }

    fn embed_documents<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, LangChainError>> {
        Box::pin(async move { self.inner.embed_documents(texts).await })
    }
}
