use std::marker::PhantomData;

use langchain_core::LangChainError;
use langchain_core::messages::{AIMessage, ToolCall};
use langchain_core::runnables::{Runnable, RunnableConfig};
use serde::de::DeserializeOwned;
use serde_json::Value;

pub use langchain_core::output_parsers::{
    JsonOutputKeyToolsParser, JsonOutputToolsParser, PydanticToolsParser,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct ToolsOutputParser;

impl ToolsOutputParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse_tool_calls(&self, message: &AIMessage) -> Vec<ToolCall> {
        extract_tool_calls(message)
    }

    pub fn parse_first_tool_call(&self, message: &AIMessage) -> Option<ToolCall> {
        self.parse_tool_calls(message).into_iter().next()
    }

    pub fn parse_tool_args(&self, message: &AIMessage) -> Vec<Value> {
        self.parse_tool_calls(message)
            .into_iter()
            .map(|tool_call| tool_call.args().clone())
            .collect()
    }

    pub fn parse_first_tool_args(&self, message: &AIMessage) -> Option<Value> {
        self.parse_tool_args(message).into_iter().next()
    }

    pub fn parse_structured<T>(&self, message: &AIMessage) -> Result<Vec<T>, LangChainError>
    where
        T: DeserializeOwned,
    {
        self.parse_tool_args(message)
            .into_iter()
            .map(|args| serde_json::from_value(args).map_err(Into::into))
            .collect()
    }

    pub fn typed<T>(&self) -> TypedToolsOutputParser<T>
    where
        T: DeserializeOwned,
    {
        TypedToolsOutputParser::new()
    }
}

impl Runnable<AIMessage, Vec<ToolCall>> for ToolsOutputParser {
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<Vec<ToolCall>, LangChainError>> {
        Box::pin(async move { Ok(self.parse_tool_calls(&input)) })
    }
}

#[derive(Debug, Clone, Default)]
pub struct TypedToolsOutputParser<T> {
    _marker: PhantomData<fn() -> T>,
}

impl<T> TypedToolsOutputParser<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Runnable<AIMessage, Vec<T>> for TypedToolsOutputParser<T>
where
    T: DeserializeOwned + Send + Sync + 'static,
{
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<Vec<T>, LangChainError>> {
        Box::pin(async move {
            input
                .tool_calls()
                .iter()
                .map(|tool_call| {
                    serde_json::from_value(tool_call.args().clone()).map_err(Into::into)
                })
                .collect()
        })
    }
}

pub fn extract_tool_calls(message: &AIMessage) -> Vec<ToolCall> {
    message.tool_calls().to_vec()
}
