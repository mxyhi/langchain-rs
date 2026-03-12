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
}

impl AIMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            response_metadata: BTreeMap::new(),
            usage_metadata: None,
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
        }
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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolMessage {
    content: String,
    tool_call_id: String,
}

impl ToolMessage {
    pub fn new(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            tool_call_id: tool_call_id.into(),
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn tool_call_id(&self) -> &str {
        &self.tool_call_id
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
        }),
        BaseMessage::System(message) => {
            json!({ "role": "system", "content": message.content() })
        }
        BaseMessage::Tool(message) => json!({
            "role": "tool",
            "content": message.content(),
            "tool_call_id": message.tool_call_id(),
        }),
    }
}

pub fn messages_to_dict(messages: &[BaseMessage]) -> Vec<Value> {
    messages.iter().map(message_to_dict).collect()
}
