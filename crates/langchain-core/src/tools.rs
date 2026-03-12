use std::any::TypeId;
use std::marker::PhantomData;
use std::sync::Arc;

use futures_util::future::BoxFuture;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::LangChainError;
use crate::documents::Document;
use crate::messages::{ToolCall, ToolMessage, ToolMessageStatus};
use crate::retrievers::BaseRetriever;
use crate::runnables::{Runnable, RunnableConfig};

pub trait BaseTool: Send + Sync {
    fn definition(&self) -> &ToolDefinition;

    fn invoke<'a>(
        &'a self,
        input: ToolCall,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<ToolMessage, LangChainError>>;
}

pub trait BaseToolkit: Send + Sync {
    fn tools(&self) -> Vec<Box<dyn BaseTool>>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InjectedToolArg;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InjectedToolCallId;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InjectedState<State>(pub PhantomData<State>);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InjectedStore<Store>(pub PhantomData<Store>);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolRuntime<State = Value, Store = Value> {
    state: State,
    store: Store,
    tool_call_id: Option<String>,
}

impl<State, Store> ToolRuntime<State, Store> {
    pub fn new(state: State, store: Store) -> Self {
        Self {
            state,
            store,
            tool_call_id: None,
        }
    }

    pub fn with_tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(tool_call_id.into());
        self
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    pub fn tool_call_id(&self) -> Option<&str> {
        self.tool_call_id.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolException {
    message: String,
}

impl ToolException {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for ToolException {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ToolException {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaAnnotationError {
    message: String,
}

impl SchemaAnnotationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for SchemaAnnotationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for SchemaAnnotationError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    name: String,
    description: String,
    parameters: Value,
    strict: Option<bool>,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: json!({
                "type": "object",
                "properties": {}
            }),
            strict: None,
        }
    }

    pub fn with_parameters(mut self, parameters: Value) -> Self {
        self.parameters = parameters;
        self
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn parameters(&self) -> &Value {
        &self.parameters
    }

    pub fn strict(&self) -> Option<bool> {
        self.strict
    }
}

pub fn tool(name: impl Into<String>, description: impl Into<String>) -> ToolDefinition {
    ToolDefinition::new(name, description)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrieverInput {
    pub query: String,
}

impl RetrieverInput {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
        }
    }

    pub fn json_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "query to look up in retriever"
                }
            },
            "required": ["query"]
        })
    }
}

type ToolHandler =
    dyn Fn(String) -> BoxFuture<'static, Result<String, LangChainError>> + Send + Sync;
type StructuredToolHandler =
    dyn Fn(Value) -> BoxFuture<'static, Result<Value, LangChainError>> + Send + Sync;

#[derive(Clone)]
pub struct Tool {
    definition: ToolDefinition,
    handler: Arc<ToolHandler>,
}

impl Tool {
    pub fn new(
        definition: ToolDefinition,
        handler: impl Fn(String) -> BoxFuture<'static, Result<String, LangChainError>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            definition,
            handler: Arc::new(handler),
        }
    }
}

impl BaseTool for Tool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    fn invoke<'a>(
        &'a self,
        input: ToolCall,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<ToolMessage, LangChainError>> {
        Box::pin(async move {
            let tool_input = extract_named_string_argument(input.args(), "input")?;
            let output = (self.handler)(tool_input).await?;
            Ok(ToolMessage::with_parts(
                output.clone(),
                input.id().unwrap_or_default(),
                Some(self.definition.name()),
                Some(Value::String(output)),
                ToolMessageStatus::Success,
            ))
        })
    }
}

#[derive(Clone)]
pub struct StructuredTool {
    definition: ToolDefinition,
    handler: Arc<StructuredToolHandler>,
}

impl StructuredTool {
    pub fn new(
        definition: ToolDefinition,
        handler: impl Fn(Value) -> BoxFuture<'static, Result<Value, LangChainError>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            definition,
            handler: Arc::new(handler),
        }
    }
}

impl BaseTool for StructuredTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    fn invoke<'a>(
        &'a self,
        input: ToolCall,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<ToolMessage, LangChainError>> {
        Box::pin(async move {
            let output = (self.handler)(input.args().clone()).await?;
            let content = match &output {
                Value::String(content) => content.clone(),
                _ => output.to_string(),
            };

            Ok(ToolMessage::with_parts(
                content,
                input.id().unwrap_or_default(),
                Some(self.definition.name()),
                Some(output),
                ToolMessageStatus::Success,
            ))
        })
    }
}

pub type ToolsRenderer = fn(&[&dyn BaseTool]) -> String;

pub fn render_text_description(tools: &[&dyn BaseTool]) -> String {
    tools
        .iter()
        .map(|tool| {
            let definition = tool.definition();
            format!("{}: {}", definition.name(), definition.description())
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn render_text_description_and_args(tools: &[&dyn BaseTool]) -> String {
    tools
        .iter()
        .map(|tool| {
            let definition = tool.definition();
            format!(
                "{}: {} | args={}",
                definition.name(),
                definition.description(),
                definition.parameters()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn convert_runnable_to_tool<R, I, O>(runnable: R, definition: ToolDefinition) -> StructuredTool
where
    R: Runnable<I, O> + Send + Sync + 'static,
    I: DeserializeOwned + Send + Sync + 'static,
    O: Serialize + Send + 'static,
{
    let runnable = Arc::new(runnable);
    let definition = normalize_runnable_tool_definition::<I>(definition);

    StructuredTool::new(definition, move |input| {
        let runnable = Arc::clone(&runnable);
        Box::pin(async move {
            let parsed_input = deserialize_runnable_input::<I>(&input)?;
            let output = runnable
                .invoke(parsed_input, RunnableConfig::default())
                .await?;
            serde_json::to_value(output).map_err(LangChainError::from)
        })
    })
}

pub fn create_retriever_tool<R>(
    retriever: R,
    name: impl Into<String>,
    description: impl Into<String>,
) -> StructuredTool
where
    R: BaseRetriever + Send + Sync + 'static,
{
    let retriever = Arc::new(retriever);
    let definition =
        ToolDefinition::new(name, description).with_parameters(RetrieverInput::json_schema());

    StructuredTool::new(definition, move |input| {
        let retriever = Arc::clone(&retriever);
        Box::pin(async move {
            let RetrieverInput { query } = deserialize_runnable_input::<RetrieverInput>(&input)?;
            let documents = retriever
                .get_relevant_documents(&query, RunnableConfig::default())
                .await?;

            // Keep the transport honest: return structured document records rather than
            // inventing a provider-specific schema.
            Ok(Value::Array(
                documents.into_iter().map(document_to_value).collect(),
            ))
        })
    })
}

fn normalize_runnable_tool_definition<I>(definition: ToolDefinition) -> ToolDefinition
where
    I: 'static,
{
    // String-input runnables still travel through ToolCall JSON, so normalize them
    // to the existing {"input": "..."} contract unless the caller supplied a schema.
    if TypeId::of::<I>() == TypeId::of::<String>()
        && definition.parameters() == &empty_object_schema()
    {
        return definition.with_parameters(single_string_input_schema());
    }

    definition
}

fn deserialize_runnable_input<I>(input: &Value) -> Result<I, LangChainError>
where
    I: DeserializeOwned + 'static,
{
    let normalized_input = if TypeId::of::<I>() == TypeId::of::<String>() {
        Value::String(extract_named_string_argument(input, "input")?)
    } else {
        input.clone()
    };

    serde_json::from_value(normalized_input).map_err(|error| {
        LangChainError::request(format!("tool input could not be deserialized: {error}"))
    })
}

fn extract_named_string_argument(input: &Value, key: &str) -> Result<String, LangChainError> {
    match input {
        Value::String(value) if key == "input" => Ok(value.clone()),
        Value::Object(map) => map
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| LangChainError::request(format!("tool input must include `{key}`"))),
        _ => Err(LangChainError::request(format!(
            "tool input must be a string or object containing `{key}`"
        ))),
    }
}

fn empty_object_schema() -> Value {
    json!({
        "type": "object",
        "properties": {}
    })
}

fn single_string_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "input": { "type": "string" }
        },
        "required": ["input"]
    })
}

fn document_to_value(document: Document) -> Value {
    json!({
        "page_content": document.page_content,
        "metadata": document.metadata,
        "id": document.id,
    })
}
