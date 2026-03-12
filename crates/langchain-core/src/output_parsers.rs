use std::marker::PhantomData;

use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

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

impl Runnable<AIMessage, Value> for JsonOutputParser {
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<Value, LangChainError>> {
        Box::pin(async move { Ok(serde_json::from_str(input.content())?) })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleJsonOutputParser;

impl SimpleJsonOutputParser {
    pub fn new() -> Self {
        Self
    }
}

impl Runnable<AIMessage, Value> for SimpleJsonOutputParser {
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<Value, LangChainError>> {
        Box::pin(async move { Ok(serde_json::from_str(input.content())?) })
    }
}

pub fn parse_openai_tool_call(raw: &Value) -> Result<ToolCall, InvalidToolCall> {
    let id = raw.get("id").and_then(Value::as_str).map(str::to_owned);
    let function = raw.get("function").cloned().unwrap_or_default();
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let raw_args = function
        .get("arguments")
        .and_then(Value::as_str)
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

    match serde_json::from_str::<Value>(&raw_args) {
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

pub fn parse_openai_tool_calls(raw_tool_calls: &[Value]) -> (Vec<ToolCall>, Vec<InvalidToolCall>) {
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

impl Runnable<AIMessage, Vec<Value>> for JsonOutputToolsParser {
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<Vec<Value>, LangChainError>> {
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

impl Runnable<AIMessage, Value> for JsonOutputKeyToolsParser {
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<Value, LangChainError>> {
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

#[derive(Debug, Clone, Default)]
pub struct PydanticOutputParser<T> {
    _marker: PhantomData<fn() -> T>,
}

impl<T> PydanticOutputParser<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Runnable<AIMessage, T> for PydanticOutputParser<T>
where
    T: DeserializeOwned + Send + Sync + 'static,
{
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<T, LangChainError>> {
        Box::pin(async move { Ok(serde_json::from_str(input.content())?) })
    }
}

#[derive(Debug, Clone, Default)]
pub struct PydanticToolsParser<T> {
    key_name: Option<String>,
    _marker: PhantomData<fn() -> T>,
}

impl<T> PydanticToolsParser<T> {
    pub fn new() -> Self {
        Self {
            key_name: None,
            _marker: PhantomData,
        }
    }

    pub fn with_key_name(mut self, key_name: impl Into<String>) -> Self {
        self.key_name = Some(key_name.into());
        self
    }
}

impl<T> Runnable<AIMessage, Vec<T>> for PydanticToolsParser<T>
where
    T: DeserializeOwned + Send + Sync + 'static,
{
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<Vec<T>, LangChainError>> {
        Box::pin(async move {
            let mut parsed = Vec::new();

            for tool_call in input.tool_calls() {
                if self
                    .key_name
                    .as_ref()
                    .is_some_and(|key_name| tool_call.name() != key_name)
                {
                    continue;
                }

                parsed.push(serde_json::from_value(tool_call.args().clone())?);
            }

            Ok(parsed)
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ListOutputParser;

impl ListOutputParser {
    pub fn new() -> Self {
        Self
    }
}

impl Runnable<AIMessage, Vec<String>> for ListOutputParser {
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<Vec<String>, LangChainError>> {
        Box::pin(async move { Ok(parse_flexible_list(input.content())) })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CommaSeparatedListOutputParser;

impl CommaSeparatedListOutputParser {
    pub fn new() -> Self {
        Self
    }
}

impl Runnable<AIMessage, Vec<String>> for CommaSeparatedListOutputParser {
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<Vec<String>, LangChainError>> {
        Box::pin(async move { Ok(parse_comma_separated_list(input.content())) })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MarkdownListOutputParser;

impl MarkdownListOutputParser {
    pub fn new() -> Self {
        Self
    }
}

impl Runnable<AIMessage, Vec<String>> for MarkdownListOutputParser {
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<Vec<String>, LangChainError>> {
        Box::pin(async move { Ok(parse_markdown_list(input.content())) })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NumberedListOutputParser;

impl NumberedListOutputParser {
    pub fn new() -> Self {
        Self
    }
}

impl Runnable<AIMessage, Vec<String>> for NumberedListOutputParser {
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<Vec<String>, LangChainError>> {
        Box::pin(async move { Ok(parse_numbered_list(input.content())) })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct XMLOutputParser;

impl XMLOutputParser {
    pub fn new() -> Self {
        Self
    }
}

impl Runnable<AIMessage, Value> for XMLOutputParser {
    fn invoke<'a>(
        &'a self,
        input: AIMessage,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<Value, LangChainError>> {
        Box::pin(async move {
            let root = parse_xml_document(input.content())?;
            let root_name = root.name.clone();
            Ok(Value::Object(Map::from_iter([(
                root_name,
                xml_node_to_value(root),
            )])))
        })
    }
}

fn parse_markdown_list(content: &str) -> Vec<String> {
    content
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            ["- ", "* ", "+ "]
                .iter()
                .find_map(|prefix| trimmed.strip_prefix(prefix))
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(str::to_owned)
        })
        .collect()
}

fn parse_numbered_list(content: &str) -> Vec<String> {
    content
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            let split_at = trimmed.find(['.', ')']).filter(|index| {
                trimmed[..*index]
                    .chars()
                    .all(|character| character.is_ascii_digit())
            })?;
            let candidate = trimmed[(split_at + 1)..].trim();
            (!candidate.is_empty()).then(|| candidate.to_owned())
        })
        .collect()
}

fn parse_comma_separated_list(content: &str) -> Vec<String> {
    content
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::to_owned)
        .collect()
}

fn parse_flexible_list(content: &str) -> Vec<String> {
    let markdown = parse_markdown_list(content);
    if !markdown.is_empty() {
        return markdown;
    }

    let numbered = parse_numbered_list(content);
    if !numbered.is_empty() {
        return numbered;
    }

    let comma_separated = parse_comma_separated_list(content);
    if comma_separated.len() > 1 {
        return comma_separated;
    }

    content
        .lines()
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::to_owned)
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct XmlNode {
    name: String,
    children: Vec<XmlNode>,
    text: String,
}

fn parse_xml_document(input: &str) -> Result<XmlNode, LangChainError> {
    let mut cursor = 0usize;
    skip_xml_whitespace(input, &mut cursor);
    let node = parse_xml_node(input, &mut cursor)?;
    skip_xml_whitespace(input, &mut cursor);
    if cursor != input.len() {
        return Err(LangChainError::request("unexpected trailing xml content"));
    }
    Ok(node)
}

fn parse_xml_node(input: &str, cursor: &mut usize) -> Result<XmlNode, LangChainError> {
    expect_char(input, cursor, '<')?;
    let name = parse_xml_name(input, cursor)?;
    expect_char(input, cursor, '>')?;

    let mut children = Vec::new();
    let mut text = String::new();

    loop {
        skip_xml_whitespace(input, cursor);

        if input[*cursor..].starts_with("</") {
            *cursor += 2;
            let closing_name = parse_xml_name(input, cursor)?;
            if closing_name != name {
                return Err(LangChainError::request(format!(
                    "xml closing tag mismatch: expected `{name}`, found `{closing_name}`"
                )));
            }
            expect_char(input, cursor, '>')?;
            break;
        }

        if input[*cursor..].starts_with('<') {
            children.push(parse_xml_node(input, cursor)?);
            continue;
        }

        let next_tag = input[*cursor..]
            .find('<')
            .ok_or_else(|| LangChainError::request("unterminated xml node"))?;
        text.push_str(&input[*cursor..(*cursor + next_tag)]);
        *cursor += next_tag;
    }

    Ok(XmlNode {
        name,
        children,
        text: text.trim().to_owned(),
    })
}

fn parse_xml_name(input: &str, cursor: &mut usize) -> Result<String, LangChainError> {
    let start = *cursor;
    while let Some(character) = input[*cursor..].chars().next() {
        if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
            *cursor += character.len_utf8();
            continue;
        }
        break;
    }

    if *cursor == start {
        return Err(LangChainError::request("xml tag name is missing"));
    }

    Ok(input[start..*cursor].to_owned())
}

fn skip_xml_whitespace(input: &str, cursor: &mut usize) {
    while let Some(character) = input[*cursor..].chars().next() {
        if character.is_whitespace() {
            *cursor += character.len_utf8();
            continue;
        }
        break;
    }
}

fn expect_char(input: &str, cursor: &mut usize, expected: char) -> Result<(), LangChainError> {
    match input[*cursor..].chars().next() {
        Some(character) if character == expected => {
            *cursor += character.len_utf8();
            Ok(())
        }
        Some(character) => Err(LangChainError::request(format!(
            "expected `{expected}` in xml, found `{character}`"
        ))),
        None => Err(LangChainError::request(format!(
            "expected `{expected}` in xml, found end of input"
        ))),
    }
}

fn xml_node_to_value(node: XmlNode) -> Value {
    if node.children.is_empty() {
        return Value::String(node.text);
    }

    let mut object = Map::new();
    for child in node.children {
        let child_name = child.name.clone();
        let child_value = xml_node_to_value(child);

        match object.get_mut(&child_name) {
            Some(Value::Array(values)) => values.push(child_value),
            Some(existing) => {
                let current = existing.clone();
                *existing = Value::Array(vec![current, child_value]);
            }
            None => {
                object.insert(child_name, child_value);
            }
        }
    }

    if !node.text.is_empty() {
        object.insert("_text".to_owned(), Value::String(node.text));
    }

    Value::Object(object)
}
