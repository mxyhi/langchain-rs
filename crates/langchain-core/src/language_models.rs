use std::collections::BTreeMap;

use futures_util::future::BoxFuture;

use crate::LangChainError;
use crate::messages::{AIMessage, BaseMessage, ResponseMetadata, UsageMetadata};
use crate::outputs::{Generation, GenerationCandidate, LLMResult};
use crate::runnables::{Runnable, RunnableConfig};

pub trait BaseChatModel: Send + Sync {
    fn model_name(&self) -> &str;

    fn generate<'a>(
        &'a self,
        messages: Vec<BaseMessage>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>>;

    fn identifying_params(&self) -> BTreeMap<String, serde_json::Value> {
        BTreeMap::new()
    }
}

impl<T> BaseChatModel for Box<T>
where
    T: BaseChatModel + ?Sized,
{
    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    fn generate<'a>(
        &'a self,
        messages: Vec<BaseMessage>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        (**self).generate(messages, config)
    }

    fn identifying_params(&self) -> BTreeMap<String, serde_json::Value> {
        (**self).identifying_params()
    }
}

impl<T> Runnable<Vec<BaseMessage>, AIMessage> for T
where
    T: BaseChatModel,
{
    fn invoke<'a>(
        &'a self,
        input: Vec<BaseMessage>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        self.generate(input, config)
    }
}

pub trait BaseLLM: Send + Sync {
    fn model_name(&self) -> &str;

    fn generate<'a>(
        &'a self,
        prompts: Vec<String>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<LLMResult, LangChainError>>;

    fn invoke_prompt<'a>(
        &'a self,
        prompt: String,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<String, LangChainError>> {
        Box::pin(async move {
            let response = self.generate(vec![prompt], config).await?;
            response
                .primary_generation()
                .map(GenerationCandidate::text)
                .map(ToOwned::to_owned)
                .ok_or_else(|| LangChainError::request("llm response contained no generations"))
        })
    }

    fn identifying_params(&self) -> BTreeMap<String, serde_json::Value> {
        BTreeMap::new()
    }
}

impl<T> BaseLLM for Box<T>
where
    T: BaseLLM + ?Sized,
{
    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    fn generate<'a>(
        &'a self,
        prompts: Vec<String>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<LLMResult, LangChainError>> {
        (**self).generate(prompts, config)
    }

    fn invoke_prompt<'a>(
        &'a self,
        prompt: String,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<String, LangChainError>> {
        (**self).invoke_prompt(prompt, config)
    }

    fn identifying_params(&self) -> BTreeMap<String, serde_json::Value> {
        (**self).identifying_params()
    }
}

impl<T> Runnable<String, String> for T
where
    T: BaseLLM,
{
    fn invoke<'a>(
        &'a self,
        input: String,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<String, LangChainError>> {
        self.invoke_prompt(input, config)
    }
}

#[derive(Debug, Clone)]
pub struct ParrotChatModel {
    model_name: String,
    parrot_buffer_length: usize,
}

impl ParrotChatModel {
    pub fn new(model_name: impl Into<String>, parrot_buffer_length: usize) -> Self {
        Self {
            model_name: model_name.into(),
            parrot_buffer_length,
        }
    }
}

impl BaseChatModel for ParrotChatModel {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn generate<'a>(
        &'a self,
        messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        Box::pin(async move {
            let last_message = messages.last().ok_or(LangChainError::EmptyMessages)?;
            let content = last_message
                .content()
                .chars()
                .take(self.parrot_buffer_length)
                .collect::<String>();
            let input_tokens = messages.iter().map(|message| message.content().len()).sum();
            let output_tokens = content.len();

            let mut metadata = ResponseMetadata::new();
            metadata.insert("model".to_owned(), self.model_name.clone().into());

            Ok(AIMessage::with_metadata(
                content,
                metadata,
                Some(UsageMetadata {
                    input_tokens,
                    output_tokens,
                    total_tokens: input_tokens + output_tokens,
                }),
            ))
        })
    }

    fn identifying_params(&self) -> BTreeMap<String, serde_json::Value> {
        BTreeMap::from([(
            "model_name".to_owned(),
            serde_json::Value::String(self.model_name.clone()),
        )])
    }
}

#[derive(Debug, Clone)]
pub struct ParrotLLM {
    model_name: String,
    parrot_buffer_length: usize,
}

impl ParrotLLM {
    pub fn new(model_name: impl Into<String>, parrot_buffer_length: usize) -> Self {
        Self {
            model_name: model_name.into(),
            parrot_buffer_length,
        }
    }
}

impl BaseLLM for ParrotLLM {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn generate<'a>(
        &'a self,
        prompts: Vec<String>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<LLMResult, LangChainError>> {
        Box::pin(async move {
            let prompt_tokens = prompts.iter().map(String::len).sum::<usize>();
            let generations = prompts
                .into_iter()
                .map(|prompt| {
                    let text = prompt
                        .chars()
                        .take(self.parrot_buffer_length)
                        .collect::<String>();
                    vec![GenerationCandidate::from(Generation::new(text))]
                })
                .collect::<Vec<_>>();
            let completion_tokens = generations
                .iter()
                .flat_map(|generation_group| generation_group.iter())
                .map(GenerationCandidate::text)
                .map(str::len)
                .sum::<usize>();

            let llm_output = BTreeMap::from([
                (
                    "model".to_owned(),
                    serde_json::Value::String(self.model_name.clone()),
                ),
                (
                    "model_name".to_owned(),
                    serde_json::Value::String(self.model_name.clone()),
                ),
                (
                    "token_usage".to_owned(),
                    serde_json::json!({
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }),
                ),
            ]);

            Ok(LLMResult::new(generations).with_output(llm_output))
        })
    }

    fn identifying_params(&self) -> BTreeMap<String, serde_json::Value> {
        BTreeMap::from([(
            "model_name".to_owned(),
            serde_json::Value::String(self.model_name.clone()),
        )])
    }
}
