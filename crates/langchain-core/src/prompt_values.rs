use crate::messages::{BaseMessage, HumanMessage};

pub trait PromptValue {
    fn to_string(&self) -> String;

    fn to_messages(&self) -> Vec<BaseMessage>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringPromptValue {
    text: String,
}

impl StringPromptValue {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn to_string(&self) -> String {
        <Self as PromptValue>::to_string(self)
    }

    pub fn to_messages(&self) -> Vec<BaseMessage> {
        <Self as PromptValue>::to_messages(self)
    }
}

impl PromptValue for StringPromptValue {
    fn to_string(&self) -> String {
        self.text.clone()
    }

    fn to_messages(&self) -> Vec<BaseMessage> {
        vec![HumanMessage::new(self.text.clone()).into()]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatPromptValue {
    messages: Vec<BaseMessage>,
}

impl ChatPromptValue {
    pub fn new(messages: Vec<BaseMessage>) -> Self {
        Self { messages }
    }

    pub fn messages(&self) -> &[BaseMessage] {
        &self.messages
    }

    pub fn to_string(&self) -> String {
        <Self as PromptValue>::to_string(self)
    }

    pub fn to_messages(&self) -> Vec<BaseMessage> {
        <Self as PromptValue>::to_messages(self)
    }
}

impl PromptValue for ChatPromptValue {
    fn to_string(&self) -> String {
        self.messages
            .iter()
            .map(|message| format!("{}: {}", message.role().as_str(), message.content()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn to_messages(&self) -> Vec<BaseMessage> {
        self.messages.clone()
    }
}

pub type ChatPromptValueConcrete = ChatPromptValue;
