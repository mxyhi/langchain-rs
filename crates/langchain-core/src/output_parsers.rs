use crate::LangChainError;
use crate::messages::AIMessage;
use crate::runnables::{Runnable, RunnableConfig};

#[derive(Debug, Clone, Copy, Default)]
pub struct StrOutputParser;

impl StrOutputParser {
    pub fn new() -> Self {
        Self
    }
}

impl Runnable<AIMessage, String> for StrOutputParser {
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<String, LangChainError>> {
        Box::pin(async move { Ok(input.content().to_owned()) })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct JsonOutputParser;

impl JsonOutputParser {
    pub fn new() -> Self {
        Self
    }
}

impl Runnable<AIMessage, serde_json::Value> for JsonOutputParser {
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<serde_json::Value, LangChainError>> {
        Box::pin(async move { Ok(serde_json::from_str(input.content())?) })
    }
}
