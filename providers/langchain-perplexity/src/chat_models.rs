use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::language_models::BaseChatModel;
use langchain_core::messages::{AIMessage, BaseMessage};
use langchain_core::runnables::RunnableConfig;

#[derive(Debug, Clone)]
pub struct ChatPerplexity {
    model: String,
}

impl ChatPerplexity {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }
}

impl BaseChatModel for ChatPerplexity {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn generate<'a>(
        &'a self,
        _messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        Box::pin(async move {
            Err(LangChainError::unsupported(
                "ChatPerplexity transport is not implemented in this milestone; use the search retriever, tools, and reasoning parsers from this crate",
            ))
        })
    }
}
