use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::messages::BaseMessage;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatSession {
    messages: Vec<BaseMessage>,
    functions: Vec<Value>,
}

impl ChatSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_messages(mut self, messages: Vec<BaseMessage>) -> Self {
        self.messages = messages;
        self
    }

    pub fn with_functions(mut self, functions: Vec<Value>) -> Self {
        self.functions = functions;
        self
    }

    pub fn messages(&self) -> &[BaseMessage] {
        &self.messages
    }

    pub fn functions(&self) -> &[Value] {
        &self.functions
    }
}
