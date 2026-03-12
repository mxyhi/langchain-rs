use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::runnables::{Runnable, RunnableConfig};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub fn strip_think_tags(text: &str) -> String {
    let mut output = String::new();
    let mut rest = text;

    loop {
        let Some(start) = rest.find("<think>") else {
            output.push_str(rest);
            break;
        };
        output.push_str(&rest[..start]);
        let after_start = &rest[start + "<think>".len()..];
        let Some(end) = after_start.find("</think>") else {
            break;
        };
        rest = &after_start[end + "</think>".len()..];
    }

    output.trim().to_owned()
}

fn extract_reasoning(text: &str) -> Option<String> {
    let start = text.find("<think>")?;
    let after_start = &text[start + "<think>".len()..];
    let end = after_start.find("</think>")?;
    Some(after_start[..end].trim().to_owned())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedReasoningOutput {
    pub reasoning: Option<String>,
    pub value: Value,
}

#[derive(Debug, Clone, Default)]
pub struct ReasoningJsonOutputParser;

impl ReasoningJsonOutputParser {
    pub fn parse_str(&self, text: &str) -> Result<Value, LangChainError> {
        let stripped = strip_think_tags(text);
        serde_json::from_str(&stripped).map_err(LangChainError::from)
    }
}

impl Runnable<String, Value> for ReasoningJsonOutputParser {
    fn invoke<'a>(
        &'a self,
        input: String,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<Value, LangChainError>> {
        Box::pin(async move { self.parse_str(&input) })
    }
}

#[derive(Debug, Clone, Default)]
pub struct ReasoningStructuredOutputParser;

impl ReasoningStructuredOutputParser {
    pub fn parse_str(&self, text: &str) -> Result<ParsedReasoningOutput, LangChainError> {
        Ok(ParsedReasoningOutput {
            reasoning: extract_reasoning(text),
            value: serde_json::from_str(&strip_think_tags(text)).map_err(LangChainError::from)?,
        })
    }
}

impl Runnable<String, ParsedReasoningOutput> for ReasoningStructuredOutputParser {
    fn invoke<'a>(
        &'a self,
        input: String,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<ParsedReasoningOutput, LangChainError>> {
        Box::pin(async move { self.parse_str(&input) })
    }
}
