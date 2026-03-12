use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::LangChainError;

pub type ResponseMetadata = BTreeMap<String, Value>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    Human,
    Ai,
    System,
    Tool,
}

impl MessageRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Human => "human",
            Self::Ai => "ai",
            Self::System => "system",
            Self::Tool => "tool",
        }
    }

    pub fn as_openai_role(self) -> &'static str {
        match self {
            Self::Human => "user",
            Self::Ai => "assistant",
            Self::System => "system",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageMetadata {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    kind: String,
    value: Value,
}

impl Annotation {
    pub fn new(kind: impl Into<String>, value: Value) -> Self {
        Self {
            kind: kind.into(),
            value,
        }
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn value(&self) -> &Value {
        &self.value
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Citation {
    url: String,
    title: Option<String>,
}

impl Citation {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            title: None,
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextContentBlock {
    text: String,
    #[serde(default)]
    annotations: Vec<Annotation>,
    #[serde(default)]
    citations: Vec<Citation>,
}

impl TextContentBlock {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            annotations: Vec::new(),
            citations: Vec::new(),
        }
    }

    pub fn with_annotations(mut self, annotations: Vec<Annotation>) -> Self {
        self.annotations = annotations;
        self
    }

    pub fn with_citations(mut self, citations: Vec<Citation>) -> Self {
        self.citations = citations;
        self
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn annotations(&self) -> &[Annotation] {
        &self.annotations
    }

    pub fn citations(&self) -> &[Citation] {
        &self.citations
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerToolCall {
    id: String,
    name: String,
    args: Value,
}

impl ServerToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, args: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            args,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn args(&self) -> &Value {
        &self.args
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ServerToolCallChunk {
    id: Option<String>,
    name: Option<String>,
    args: Option<String>,
}

impl ServerToolCallChunk {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_args(mut self, args: impl Into<String>) -> Self {
        self.args = Some(args.into());
        self
    }

    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn args(&self) -> Option<&str> {
        self.args.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerToolResult {
    id: String,
    name: Option<String>,
    result: Value,
    is_error: bool,
}

impl ServerToolResult {
    pub fn new(id: impl Into<String>, result: Value) -> Self {
        Self {
            id: id.into(),
            name: None,
            result,
            is_error: false,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_error(mut self, is_error: bool) -> Self {
        self.is_error = is_error;
        self
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn result(&self) -> &Value {
        &self.result
    }

    pub fn is_error(&self) -> bool {
        self.is_error
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ContentBlock {
    Text(TextContentBlock),
    ServerToolCall(ServerToolCall),
    ServerToolResult(ServerToolResult),
    NonStandard(Value),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    id: Option<String>,
    name: String,
    args: Value,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, args: Value) -> Self {
        Self {
            id: None,
            name: name.into(),
            args,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn args(&self) -> &Value {
        &self.args
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ToolCallChunk {
    id: Option<String>,
    name: Option<String>,
    args: Option<String>,
}

impl ToolCallChunk {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_args(mut self, args: impl Into<String>) -> Self {
        self.args = Some(args.into());
        self
    }

    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn args(&self) -> Option<&str> {
        self.args.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvalidToolCall {
    id: Option<String>,
    name: Option<String>,
    raw_args: Option<String>,
    error: Option<String>,
}

impl InvalidToolCall {
    pub fn new(
        name: Option<impl Into<String>>,
        raw_args: Option<impl Into<String>>,
        error: Option<impl Into<String>>,
    ) -> Self {
        Self {
            id: None,
            name: name.map(Into::into),
            raw_args: raw_args.map(Into::into),
            error: error.map(Into::into),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn raw_args(&self) -> Option<&str> {
        self.raw_args.as_deref()
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanMessage {
    content: String,
}

impl HumanMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemMessage {
    content: String,
}

impl SystemMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    role: String,
    content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn role(&self) -> &str {
        &self.role
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionMessage {
    name: String,
    content: String,
}

impl FunctionMessage {
    pub fn new(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content: content.into(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoveMessage {
    id: String,
}

impl RemoveMessage {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }

    pub fn id(&self) -> &str {
        &self.id
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AIMessage {
    content: String,
    response_metadata: ResponseMetadata,
    usage_metadata: Option<UsageMetadata>,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
    #[serde(default)]
    invalid_tool_calls: Vec<InvalidToolCall>,
}

impl AIMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            response_metadata: BTreeMap::new(),
            usage_metadata: None,
            tool_calls: Vec::new(),
            invalid_tool_calls: Vec::new(),
        }
    }

    pub fn with_metadata(
        content: impl Into<String>,
        response_metadata: ResponseMetadata,
        usage_metadata: Option<UsageMetadata>,
    ) -> Self {
        Self {
            content: content.into(),
            response_metadata,
            usage_metadata,
            tool_calls: Vec::new(),
            invalid_tool_calls: Vec::new(),
        }
    }

    pub fn with_parts(
        content: impl Into<String>,
        response_metadata: ResponseMetadata,
        usage_metadata: Option<UsageMetadata>,
        tool_calls: Vec<ToolCall>,
        invalid_tool_calls: Vec<InvalidToolCall>,
    ) -> Self {
        Self {
            content: content.into(),
            response_metadata,
            usage_metadata,
            tool_calls,
            invalid_tool_calls,
        }
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    pub fn with_invalid_tool_calls(mut self, invalid_tool_calls: Vec<InvalidToolCall>) -> Self {
        self.invalid_tool_calls = invalid_tool_calls;
        self
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn response_metadata(&self) -> &ResponseMetadata {
        &self.response_metadata
    }

    pub fn usage_metadata(&self) -> Option<&UsageMetadata> {
        self.usage_metadata.as_ref()
    }

    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    pub fn invalid_tool_calls(&self) -> &[InvalidToolCall] {
        &self.invalid_tool_calls
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanMessageChunk {
    content: String,
}

impl HumanMessageChunk {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemMessageChunk {
    content: String,
}

impl SystemMessageChunk {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AIMessageChunk {
    content: String,
    response_metadata: ResponseMetadata,
    usage_metadata: Option<UsageMetadata>,
    #[serde(default)]
    tool_call_chunks: Vec<ToolCallChunk>,
}

impl AIMessageChunk {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            response_metadata: BTreeMap::new(),
            usage_metadata: None,
            tool_call_chunks: Vec::new(),
        }
    }

    pub fn with_response_metadata(mut self, response_metadata: ResponseMetadata) -> Self {
        self.response_metadata = response_metadata;
        self
    }

    pub fn with_usage_metadata(mut self, usage_metadata: UsageMetadata) -> Self {
        self.usage_metadata = Some(usage_metadata);
        self
    }

    pub fn with_tool_call_chunks(mut self, tool_call_chunks: Vec<ToolCallChunk>) -> Self {
        self.tool_call_chunks = tool_call_chunks;
        self
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn response_metadata(&self) -> &ResponseMetadata {
        &self.response_metadata
    }

    pub fn usage_metadata(&self) -> Option<&UsageMetadata> {
        self.usage_metadata.as_ref()
    }

    pub fn tool_call_chunks(&self) -> &[ToolCallChunk] {
        &self.tool_call_chunks
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolMessageStatus {
    Success,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolMessage {
    content: String,
    tool_call_id: String,
    name: Option<String>,
    artifact: Option<Value>,
    status: ToolMessageStatus,
}

impl ToolMessage {
    pub fn new(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            tool_call_id: tool_call_id.into(),
            name: None,
            artifact: None,
            status: ToolMessageStatus::Success,
        }
    }

    pub fn with_parts(
        content: impl Into<String>,
        tool_call_id: impl Into<String>,
        name: Option<impl Into<String>>,
        artifact: Option<Value>,
        status: ToolMessageStatus,
    ) -> Self {
        Self {
            content: content.into(),
            tool_call_id: tool_call_id.into(),
            name: name.map(Into::into),
            artifact,
            status,
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn tool_call_id(&self) -> &str {
        &self.tool_call_id
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn artifact(&self) -> Option<&Value> {
        self.artifact.as_ref()
    }

    pub fn status(&self) -> ToolMessageStatus {
        self.status
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolMessageChunk {
    content: String,
    tool_call_id: String,
    name: Option<String>,
    artifact: Option<Value>,
    status: ToolMessageStatus,
}

impl ToolMessageChunk {
    pub fn new(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            tool_call_id: tool_call_id.into(),
            name: None,
            artifact: None,
            status: ToolMessageStatus::Success,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_artifact(mut self, artifact: Value) -> Self {
        self.artifact = Some(artifact);
        self
    }

    pub fn with_status(mut self, status: ToolMessageStatus) -> Self {
        self.status = status;
        self
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn tool_call_id(&self) -> &str {
        &self.tool_call_id
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn artifact(&self) -> Option<&Value> {
        self.artifact.as_ref()
    }

    pub fn status(&self) -> ToolMessageStatus {
        self.status
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum BaseMessage {
    Human(HumanMessage),
    Ai(AIMessage),
    System(SystemMessage),
    Tool(ToolMessage),
}

impl BaseMessage {
    pub fn role(&self) -> MessageRole {
        match self {
            Self::Human(_) => MessageRole::Human,
            Self::Ai(_) => MessageRole::Ai,
            Self::System(_) => MessageRole::System,
            Self::Tool(_) => MessageRole::Tool,
        }
    }

    pub fn content(&self) -> &str {
        match self {
            Self::Human(message) => message.content(),
            Self::Ai(message) => message.content(),
            Self::System(message) => message.content(),
            Self::Tool(message) => message.content(),
        }
    }
}

impl From<HumanMessage> for BaseMessage {
    fn from(value: HumanMessage) -> Self {
        Self::Human(value)
    }
}

impl From<AIMessage> for BaseMessage {
    fn from(value: AIMessage) -> Self {
        Self::Ai(value)
    }
}

impl From<SystemMessage> for BaseMessage {
    fn from(value: SystemMessage) -> Self {
        Self::System(value)
    }
}

impl From<ToolMessage> for BaseMessage {
    fn from(value: ToolMessage) -> Self {
        Self::Tool(value)
    }
}

pub type AnyMessage = BaseMessage;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum BaseMessageChunk {
    Human(HumanMessageChunk),
    Ai(AIMessageChunk),
    System(SystemMessageChunk),
    Tool(ToolMessageChunk),
}

#[derive(Debug, Clone, PartialEq)]
pub enum MessageLikeRepresentation {
    BaseMessage(BaseMessage),
    Text(String),
    RoleAndContent { role: String, content: String },
    Chat(ChatMessage),
    Function(FunctionMessage),
    Dict(Value),
}

impl From<BaseMessage> for MessageLikeRepresentation {
    fn from(value: BaseMessage) -> Self {
        Self::BaseMessage(value)
    }
}

impl From<HumanMessage> for MessageLikeRepresentation {
    fn from(value: HumanMessage) -> Self {
        Self::BaseMessage(BaseMessage::from(value))
    }
}

impl From<AIMessage> for MessageLikeRepresentation {
    fn from(value: AIMessage) -> Self {
        Self::BaseMessage(BaseMessage::from(value))
    }
}

impl From<SystemMessage> for MessageLikeRepresentation {
    fn from(value: SystemMessage) -> Self {
        Self::BaseMessage(BaseMessage::from(value))
    }
}

impl From<ToolMessage> for MessageLikeRepresentation {
    fn from(value: ToolMessage) -> Self {
        Self::BaseMessage(BaseMessage::from(value))
    }
}

impl From<ChatMessage> for MessageLikeRepresentation {
    fn from(value: ChatMessage) -> Self {
        Self::Chat(value)
    }
}

impl From<FunctionMessage> for MessageLikeRepresentation {
    fn from(value: FunctionMessage) -> Self {
        Self::Function(value)
    }
}

impl From<String> for MessageLikeRepresentation {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for MessageLikeRepresentation {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

pub fn trim_messages(messages: &[BaseMessage], max_messages: usize) -> Vec<BaseMessage> {
    let start = messages.len().saturating_sub(max_messages);
    messages[start..].to_vec()
}

pub fn filter_messages(messages: &[BaseMessage], roles: &[MessageRole]) -> Vec<BaseMessage> {
    if roles.is_empty() {
        return messages.to_vec();
    }

    messages
        .iter()
        .filter(|message| roles.contains(&message.role()))
        .cloned()
        .collect()
}

pub fn merge_message_runs(messages: &[BaseMessage]) -> Vec<BaseMessage> {
    let mut merged = Vec::new();

    for message in messages {
        if let Some(previous) = merged.last_mut()
            && try_merge_base_messages(previous, message)
        {
            continue;
        }

        merged.push(message.clone());
    }

    merged
}

pub fn message_chunk_to_message(chunk: &BaseMessageChunk) -> BaseMessage {
    match chunk {
        BaseMessageChunk::Human(chunk) => BaseMessage::from(HumanMessage::new(chunk.content())),
        BaseMessageChunk::System(chunk) => BaseMessage::from(SystemMessage::new(chunk.content())),
        BaseMessageChunk::Tool(chunk) => BaseMessage::from(ToolMessage::with_parts(
            chunk.content(),
            chunk.tool_call_id(),
            chunk.name(),
            chunk.artifact().cloned(),
            chunk.status(),
        )),
        BaseMessageChunk::Ai(chunk) => {
            let (tool_calls, invalid_tool_calls) = chunk
                .tool_call_chunks()
                .iter()
                .map(tool_call_from_chunk)
                .fold((Vec::new(), Vec::new()), |mut acc, item| {
                    match item {
                        Ok(tool_call) => acc.0.push(tool_call),
                        Err(invalid_tool_call) => acc.1.push(invalid_tool_call),
                    }
                    acc
                });

            BaseMessage::from(AIMessage::with_parts(
                chunk.content(),
                chunk.response_metadata().clone(),
                chunk.usage_metadata().cloned(),
                tool_calls,
                invalid_tool_calls,
            ))
        }
    }
}

pub fn message_to_dict(message: &BaseMessage) -> Value {
    match message {
        BaseMessage::Human(message) => {
            json!({ "role": "human", "content": message.content() })
        }
        BaseMessage::Ai(message) => json!({
            "role": "ai",
            "content": message.content(),
            "response_metadata": message.response_metadata(),
            "usage_metadata": message.usage_metadata(),
            "tool_calls": message.tool_calls(),
            "invalid_tool_calls": message.invalid_tool_calls(),
        }),
        BaseMessage::System(message) => {
            json!({ "role": "system", "content": message.content() })
        }
        BaseMessage::Tool(message) => json!({
            "role": "tool",
            "content": message.content(),
            "tool_call_id": message.tool_call_id(),
            "name": message.name(),
            "artifact": message.artifact(),
            "status": message.status(),
        }),
    }
}

pub fn messages_to_dict(messages: &[BaseMessage]) -> Vec<Value> {
    messages.iter().map(message_to_dict).collect()
}

pub fn messages_from_dict(messages: &[Value]) -> Result<Vec<BaseMessage>, LangChainError> {
    messages.iter().map(message_from_dict).collect()
}

pub fn convert_to_messages<I>(messages: I) -> Result<Vec<BaseMessage>, LangChainError>
where
    I: IntoIterator<Item = MessageLikeRepresentation>,
{
    messages
        .into_iter()
        .map(message_like_to_message)
        .collect::<Result<Vec<_>, _>>()
}

pub fn convert_to_openai_messages(messages: &[BaseMessage]) -> Vec<Value> {
    messages
        .iter()
        .map(|message| match message {
            BaseMessage::Human(message) => {
                json!({ "role": "user", "content": message.content() })
            }
            BaseMessage::Ai(message) => {
                let mut body = Map::from_iter([
                    ("role".to_owned(), Value::String("assistant".to_owned())),
                    (
                        "content".to_owned(),
                        Value::String(message.content().to_owned()),
                    ),
                ]);

                if !message.tool_calls().is_empty() {
                    body.insert(
                        "tool_calls".to_owned(),
                        Value::Array(
                            message
                                .tool_calls()
                                .iter()
                                .map(|tool_call| {
                                    json!({
                                        "id": tool_call.id(),
                                        "type": "function",
                                        "function": {
                                            "name": tool_call.name(),
                                            "arguments": tool_call.args().to_string(),
                                        }
                                    })
                                })
                                .collect(),
                        ),
                    );
                }

                Value::Object(body)
            }
            BaseMessage::System(message) => {
                json!({ "role": "system", "content": message.content() })
            }
            BaseMessage::Tool(message) => json!({
                "role": "tool",
                "content": message.content(),
                "tool_call_id": message.tool_call_id(),
                "name": message.name(),
            }),
        })
        .collect()
}

fn message_from_dict(raw: &Value) -> Result<BaseMessage, LangChainError> {
    let Some(object) = raw.as_object() else {
        return Err(LangChainError::request("message must be a JSON object"));
    };

    let role = string_field(object, "role")?;
    let content = string_field(object, "content")?;

    match normalize_role(role) {
        Some(MessageRole::Human) => Ok(BaseMessage::from(HumanMessage::new(content))),
        Some(MessageRole::Ai) => Ok(BaseMessage::from(AIMessage::with_parts(
            content,
            optional_object_field(object, "response_metadata")?,
            optional_typed_field(object, "usage_metadata")?,
            optional_typed_field(object, "tool_calls")?.unwrap_or_default(),
            optional_typed_field(object, "invalid_tool_calls")?.unwrap_or_default(),
        ))),
        Some(MessageRole::System) => Ok(BaseMessage::from(SystemMessage::new(content))),
        Some(MessageRole::Tool) => {
            let tool_call_id = string_field(object, "tool_call_id")?;
            let name = optional_string_field(object, "name");
            let artifact = object.get("artifact").cloned();
            let status =
                optional_typed_field(object, "status")?.unwrap_or(ToolMessageStatus::Success);

            Ok(BaseMessage::from(ToolMessage::with_parts(
                content,
                tool_call_id,
                name,
                artifact,
                status,
            )))
        }
        None => Err(LangChainError::request(format!(
            "unsupported message role `{role}`"
        ))),
    }
}

fn message_like_to_message(
    message: MessageLikeRepresentation,
) -> Result<BaseMessage, LangChainError> {
    match message {
        MessageLikeRepresentation::BaseMessage(message) => Ok(message),
        MessageLikeRepresentation::Text(content) => {
            Ok(BaseMessage::from(HumanMessage::new(content)))
        }
        MessageLikeRepresentation::RoleAndContent { role, content } => {
            message_from_role_and_content(&role, content)
        }
        MessageLikeRepresentation::Chat(message) => {
            message_from_role_and_content(message.role(), message.content().to_owned())
        }
        MessageLikeRepresentation::Function(message) => {
            Ok(BaseMessage::from(ToolMessage::with_parts(
                message.content(),
                message.name(),
                Some(message.name()),
                None,
                ToolMessageStatus::Success,
            )))
        }
        MessageLikeRepresentation::Dict(raw) => message_from_dict(&raw),
    }
}

fn message_from_role_and_content(
    role: &str,
    content: String,
) -> Result<BaseMessage, LangChainError> {
    match normalize_role(role) {
        Some(MessageRole::Human) => Ok(BaseMessage::from(HumanMessage::new(content))),
        Some(MessageRole::Ai) => Ok(BaseMessage::from(AIMessage::new(content))),
        Some(MessageRole::System) => Ok(BaseMessage::from(SystemMessage::new(content))),
        Some(MessageRole::Tool) => Ok(BaseMessage::from(ToolMessage::new(content, ""))),
        None => Err(LangChainError::request(format!(
            "unsupported message role `{role}`"
        ))),
    }
}

fn tool_call_from_chunk(chunk: &ToolCallChunk) -> Result<ToolCall, InvalidToolCall> {
    let Some(name) = chunk.name().map(str::to_owned) else {
        let mut invalid = InvalidToolCall::new(
            None::<String>,
            chunk.args().map(str::to_owned),
            Some("tool call chunk is missing name"),
        );
        if let Some(id) = chunk.id() {
            invalid = invalid.with_id(id);
        }
        return Err(invalid);
    };

    let Some(raw_args) = chunk.args().map(str::to_owned) else {
        let mut invalid = InvalidToolCall::new(
            Some(name),
            None::<String>,
            Some("tool call chunk is missing args"),
        );
        if let Some(id) = chunk.id() {
            invalid = invalid.with_id(id);
        }
        return Err(invalid);
    };

    match serde_json::from_str::<Value>(&raw_args) {
        Ok(args) => {
            let mut tool_call = ToolCall::new(name, args);
            if let Some(id) = chunk.id() {
                tool_call = tool_call.with_id(id);
            }
            Ok(tool_call)
        }
        Err(error) => {
            let mut invalid =
                InvalidToolCall::new(Some(name), Some(raw_args), Some(error.to_string()));
            if let Some(id) = chunk.id() {
                invalid = invalid.with_id(id);
            }
            Err(invalid)
        }
    }
}

fn try_merge_base_messages(target: &mut BaseMessage, incoming: &BaseMessage) -> bool {
    match (target, incoming) {
        (BaseMessage::Human(target), BaseMessage::Human(incoming)) => {
            merge_plain_text(&mut target.content, incoming.content());
            true
        }
        (BaseMessage::System(target), BaseMessage::System(incoming)) => {
            merge_plain_text(&mut target.content, incoming.content());
            true
        }
        (BaseMessage::Ai(target), BaseMessage::Ai(incoming)) => {
            // This keeps consecutive assistant generations readable while preserving tool metadata.
            merge_plain_text(&mut target.content, incoming.content());
            target
                .response_metadata
                .extend(incoming.response_metadata.clone());
            if incoming.usage_metadata.is_some() {
                target.usage_metadata = incoming.usage_metadata.clone();
            }
            target.tool_calls.extend(incoming.tool_calls.clone());
            target
                .invalid_tool_calls
                .extend(incoming.invalid_tool_calls.clone());
            true
        }
        (BaseMessage::Tool(target), BaseMessage::Tool(incoming))
            if target.tool_call_id == incoming.tool_call_id
                && target.name == incoming.name
                && target.status == incoming.status
                && target.artifact == incoming.artifact =>
        {
            merge_plain_text(&mut target.content, incoming.content());
            true
        }
        _ => false,
    }
}

fn merge_plain_text(target: &mut String, incoming: &str) {
    if incoming.is_empty() {
        return;
    }

    if target.is_empty() {
        *target = incoming.to_owned();
        return;
    }

    target.push('\n');
    target.push_str(incoming);
}

fn normalize_role(role: &str) -> Option<MessageRole> {
    match role {
        "human" | "user" => Some(MessageRole::Human),
        "ai" | "assistant" => Some(MessageRole::Ai),
        "system" => Some(MessageRole::System),
        "tool" | "function" => Some(MessageRole::Tool),
        _ => None,
    }
}

fn string_field<'a>(object: &'a Map<String, Value>, name: &str) -> Result<&'a str, LangChainError> {
    object
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| LangChainError::request(format!("message is missing string field `{name}`")))
}

fn optional_string_field(object: &Map<String, Value>, name: &str) -> Option<String> {
    object.get(name).and_then(Value::as_str).map(str::to_owned)
}

fn optional_object_field(
    object: &Map<String, Value>,
    name: &str,
) -> Result<ResponseMetadata, LangChainError> {
    match object.get(name) {
        Some(Value::Object(value)) => Ok(value
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()),
        Some(Value::Null) | None => Ok(BTreeMap::new()),
        Some(_) => Err(LangChainError::request(format!(
            "message field `{name}` must be an object"
        ))),
    }
}

fn optional_typed_field<T>(
    object: &Map<String, Value>,
    name: &str,
) -> Result<Option<T>, LangChainError>
where
    T: for<'de> Deserialize<'de>,
{
    match object.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => Ok(Some(serde_json::from_value(value.clone())?)),
    }
}
