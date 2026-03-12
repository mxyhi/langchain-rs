use std::sync::RwLock;

use futures_util::future::BoxFuture;

use crate::messages::{AIMessage, BaseMessage, HumanMessage};

pub trait BaseChatMessageHistory: Send + Sync {
    fn messages(&self) -> Vec<BaseMessage>;

    fn add_message(&self, message: BaseMessage);

    fn clear(&self);

    fn add_messages(&self, messages: Vec<BaseMessage>) {
        for message in messages {
            self.add_message(message);
        }
    }

    fn add_user_message(&self, message: impl Into<String>) {
        self.add_message(HumanMessage::new(message).into());
    }

    fn add_ai_message(&self, message: impl Into<String>) {
        self.add_message(AIMessage::new(message).into());
    }

    fn aget_messages<'a>(&'a self) -> BoxFuture<'a, Vec<BaseMessage>> {
        Box::pin(async move { self.messages() })
    }

    fn aadd_messages<'a>(&'a self, messages: Vec<BaseMessage>) -> BoxFuture<'a, ()> {
        Box::pin(async move { self.add_messages(messages) })
    }

    fn aclear<'a>(&'a self) -> BoxFuture<'a, ()> {
        Box::pin(async move { self.clear() })
    }
}

#[derive(Debug, Default)]
pub struct InMemoryChatMessageHistory {
    messages: RwLock<Vec<BaseMessage>>,
}

impl InMemoryChatMessageHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn messages(&self) -> Vec<BaseMessage> {
        <Self as BaseChatMessageHistory>::messages(self)
    }

    pub fn add_message(&self, message: BaseMessage) {
        <Self as BaseChatMessageHistory>::add_message(self, message);
    }

    pub fn add_messages(&self, messages: Vec<BaseMessage>) {
        <Self as BaseChatMessageHistory>::add_messages(self, messages);
    }

    pub fn add_user_message(&self, message: impl Into<String>) {
        <Self as BaseChatMessageHistory>::add_user_message(self, message);
    }

    pub fn add_ai_message(&self, message: impl Into<String>) {
        <Self as BaseChatMessageHistory>::add_ai_message(self, message);
    }

    pub fn aget_messages<'a>(&'a self) -> BoxFuture<'a, Vec<BaseMessage>> {
        <Self as BaseChatMessageHistory>::aget_messages(self)
    }

    pub fn clear(&self) {
        <Self as BaseChatMessageHistory>::clear(self);
    }
}

impl BaseChatMessageHistory for InMemoryChatMessageHistory {
    fn messages(&self) -> Vec<BaseMessage> {
        self.messages
            .read()
            .expect("chat history read lock poisoned")
            .clone()
    }

    fn add_message(&self, message: BaseMessage) {
        self.messages
            .write()
            .expect("chat history write lock poisoned")
            .push(message);
    }

    fn clear(&self) {
        self.messages
            .write()
            .expect("chat history write lock poisoned")
            .clear();
    }
}
