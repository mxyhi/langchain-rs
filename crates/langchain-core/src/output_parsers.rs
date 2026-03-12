use crate::LangChainError;
use crate::messages::{AIMessage, InvalidToolCall, ToolCall};
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

pub fn parse_openai_tool_call(raw: &serde_json::Value) -> Result<ToolCall, InvalidToolCall> {
    let id = raw
        .get("id")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let function = raw.get("function").cloned().unwrap_or_default();
    let name = function
        .get("name")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let raw_args = function
        .get("arguments")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);

    let Some(name) = name else {
        let mut invalid = InvalidToolCall::new(
            None::<String>,
            raw_args,
            Some("tool call is missing function.name"),
        );
        if let Some(id) = id {
            invalid = invalid.with_id(id);
        }
        return Err(invalid);
    };

    let Some(raw_args) = raw_args else {
        let mut invalid = InvalidToolCall::new(
            Some(name),
            None::<String>,
            Some("tool call is missing function.arguments"),
        );
        if let Some(id) = id {
            invalid = invalid.with_id(id);
        }
        return Err(invalid);
    };

    match serde_json::from_str::<serde_json::Value>(&raw_args) {
        Ok(args) => {
            let mut tool_call = ToolCall::new(name, args);
            if let Some(id) = id {
                tool_call = tool_call.with_id(id);
            }
            Ok(tool_call)
        }
        Err(error) => {
            let mut invalid =
                InvalidToolCall::new(Some(name), Some(raw_args), Some(error.to_string()));
            if let Some(id) = id {
                invalid = invalid.with_id(id);
            }
            Err(invalid)
        }
    }
}

pub fn parse_openai_tool_calls(
    raw_tool_calls: &[serde_json::Value],
) -> (Vec<ToolCall>, Vec<InvalidToolCall>) {
    let mut parsed = Vec::new();
    let mut invalid = Vec::new();

    for raw_tool_call in raw_tool_calls {
        match parse_openai_tool_call(raw_tool_call) {
            Ok(tool_call) => parsed.push(tool_call),
            Err(invalid_tool_call) => invalid.push(invalid_tool_call),
        }
    }

    (parsed, invalid)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct JsonOutputToolsParser;

impl JsonOutputToolsParser {
    pub fn new() -> Self {
        Self
    }
}

impl Runnable<AIMessage, Vec<serde_json::Value>> for JsonOutputToolsParser {
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<Vec<serde_json::Value>, LangChainError>> {
        Box::pin(async move {
            Ok(input
                .tool_calls()
                .iter()
                .map(|tool_call| tool_call.args().clone())
                .collect())
        })
    }
}

#[derive(Debug, Clone)]
pub struct JsonOutputKeyToolsParser {
    key_name: String,
}

impl JsonOutputKeyToolsParser {
    pub fn new(key_name: impl Into<String>) -> Self {
        Self {
            key_name: key_name.into(),
        }
    }
}

impl Runnable<AIMessage, serde_json::Value> for JsonOutputKeyToolsParser {
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<serde_json::Value, LangChainError>> {
        Box::pin(async move {
            input
                .tool_calls()
                .iter()
                .find(|tool_call| tool_call.name() == self.key_name)
                .map(|tool_call| tool_call.args().clone())
                .ok_or_else(|| {
                    LangChainError::request(format!(
                        "tool call `{}` not present in AI message",
                        self.key_name
                    ))
                })
        })
    }
}
