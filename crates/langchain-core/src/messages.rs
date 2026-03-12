use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

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

pub fn trim_messages(messages: &[BaseMessage], max_messages: usize) -> Vec<BaseMessage> {
    let start = messages.len().saturating_sub(max_messages);
    messages[start..].to_vec()
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
