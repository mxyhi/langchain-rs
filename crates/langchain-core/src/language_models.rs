use std::collections::BTreeMap;

use futures_util::future::BoxFuture;

use crate::LangChainError;
use crate::messages::{AIMessage, BaseMessage, ResponseMetadata, UsageMetadata};
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

impl<T> Runnable<Vec<BaseMessage>, AIMessage> for T
where
    T: BaseChatModel,
{
    fn invoke<'a>(
        &'a self,
        input: Vec<BaseMessage>,
        config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<AIMessage, LangChainError>> {
        self.generate(input, config)
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
