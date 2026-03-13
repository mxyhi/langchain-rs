use langchain_core::LangChainError;
use langchain_core::messages::{BaseMessage, convert_to_openai_messages, messages_from_dict};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const CLASSIC_PACKAGE: &str = crate::_api::CLASSIC_PACKAGE;

pub mod openai {
    use super::*;

    pub const CLASSIC_PACKAGE: &str = crate::_api::CLASSIC_PACKAGE;

    pub type IndexableBaseModel = Value;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Choice {
        pub index: usize,
        pub message: Value,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ChoiceChunk {
        pub index: usize,
        pub delta: Value,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ChatCompletion {
        pub id: String,
        pub choices: Vec<Choice>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ChatCompletionChunk {
        pub id: String,
        pub choices: Vec<ChoiceChunk>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ChatCompletions {
        pub completions: Vec<ChatCompletion>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Completions {
        pub data: Vec<Value>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Chat {
        pub messages: Vec<Value>,
    }

    pub fn convert_message_to_dict(message: &BaseMessage) -> Value {
        convert_to_openai_messages(std::slice::from_ref(message))
            .into_iter()
            .next()
            .unwrap_or_else(|| json!({}))
    }

    pub fn convert_dict_to_message(message: &Value) -> Result<BaseMessage, LangChainError> {
        messages_from_dict(std::slice::from_ref(message))?
            .into_iter()
            .next()
            .ok_or_else(|| LangChainError::request("openai adapter input contained no message"))
    }

    pub fn convert_openai_messages(
        messages: Vec<Value>,
    ) -> Result<Vec<BaseMessage>, LangChainError> {
        messages_from_dict(&messages)
    }

    pub fn convert_messages_for_finetuning(messages: &[BaseMessage]) -> Vec<Value> {
        messages.iter().map(convert_message_to_dict).collect()
    }

    pub fn chat(messages: &[BaseMessage]) -> Chat {
        Chat {
            messages: convert_messages_for_finetuning(messages),
        }
    }
}
