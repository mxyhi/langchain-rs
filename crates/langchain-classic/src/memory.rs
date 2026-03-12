use std::collections::{BTreeMap, BTreeSet};
use std::sync::RwLock;

use langchain_core::messages::{
    AIMessage, BaseMessage, HumanMessage, MessageLikeRepresentation, get_buffer_string,
    messages_to_dict,
};
use serde_json::{Value, json};

use crate::LangChainError;
use crate::base_memory::BaseMemory;

pub mod prompt;

pub use langchain_core::chat_history::{BaseChatMessageHistory, InMemoryChatMessageHistory};

const DEFAULT_MEMORY_KEY: &str = "history";
const DEFAULT_OUTPUT_KEY: &str = "output";

#[derive(Debug)]
pub struct ConversationBufferMemory {
    chat_memory: InMemoryChatMessageHistory,
    input_key: Option<String>,
    output_key: Option<String>,
    return_messages: bool,
    memory_key: String,
}

impl Default for ConversationBufferMemory {
    fn default() -> Self {
        Self {
            chat_memory: InMemoryChatMessageHistory::new(),
            input_key: None,
            output_key: None,
            return_messages: false,
            memory_key: DEFAULT_MEMORY_KEY.to_owned(),
        }
    }
}

impl ConversationBufferMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_input_key(mut self, input_key: impl Into<String>) -> Self {
        self.input_key = Some(input_key.into());
        self
    }

    pub fn with_output_key(mut self, output_key: impl Into<String>) -> Self {
        self.output_key = Some(output_key.into());
        self
    }

    pub fn with_return_messages(mut self, return_messages: bool) -> Self {
        self.return_messages = return_messages;
        self
    }

    pub fn with_memory_key(mut self, memory_key: impl Into<String>) -> Self {
        self.memory_key = memory_key.into();
        self
    }

    pub fn buffer_as_messages(&self) -> Vec<BaseMessage> {
        self.chat_memory.messages()
    }

    pub fn buffer_as_str(&self) -> Result<String, LangChainError> {
        get_buffer_string(
            self.buffer_as_messages()
                .into_iter()
                .map(MessageLikeRepresentation::from),
        )
    }

    pub fn try_save_context(
        &self,
        inputs: BTreeMap<String, Value>,
        outputs: BTreeMap<String, Value>,
    ) -> Result<(), LangChainError> {
        let (input, output) = resolve_input_output(
            &inputs,
            &outputs,
            self.input_key.as_deref(),
            self.output_key.as_deref(),
            &[self.memory_key.clone()],
        )?;

        self.chat_memory.add_messages(vec![
            HumanMessage::new(input).into(),
            AIMessage::new(output).into(),
        ]);
        Ok(())
    }
}

impl BaseMemory for ConversationBufferMemory {
    fn memory_variables(&self) -> Vec<String> {
        vec![self.memory_key.clone()]
    }

    fn load_memory_variables(&self, _inputs: BTreeMap<String, Value>) -> BTreeMap<String, Value> {
        let value = if self.return_messages {
            json!(messages_to_dict(&self.buffer_as_messages()))
        } else {
            json!(
                self.buffer_as_str()
                    .expect("conversation buffer memory should render to a string")
            )
        };

        [(self.memory_key.clone(), value)].into()
    }

    fn save_context(&self, inputs: BTreeMap<String, Value>, outputs: BTreeMap<String, Value>) {
        self.try_save_context(inputs, outputs)
            .expect("conversation buffer memory should resolve input and output keys")
    }

    fn clear(&self) {
        self.chat_memory.clear();
    }
}

#[derive(Debug)]
pub struct ConversationStringBufferMemory {
    human_prefix: String,
    ai_prefix: String,
    buffer: RwLock<String>,
    output_key: Option<String>,
    input_key: Option<String>,
    memory_key: String,
}

impl Default for ConversationStringBufferMemory {
    fn default() -> Self {
        Self {
            human_prefix: "Human".to_owned(),
            ai_prefix: "AI".to_owned(),
            buffer: RwLock::new(String::new()),
            output_key: None,
            input_key: None,
            memory_key: DEFAULT_MEMORY_KEY.to_owned(),
        }
    }
}

impl ConversationStringBufferMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_input_key(mut self, input_key: impl Into<String>) -> Self {
        self.input_key = Some(input_key.into());
        self
    }

    pub fn with_output_key(mut self, output_key: impl Into<String>) -> Self {
        self.output_key = Some(output_key.into());
        self
    }

    pub fn with_memory_key(mut self, memory_key: impl Into<String>) -> Self {
        self.memory_key = memory_key.into();
        self
    }

    pub fn with_human_prefix(mut self, human_prefix: impl Into<String>) -> Self {
        self.human_prefix = human_prefix.into();
        self
    }

    pub fn with_ai_prefix(mut self, ai_prefix: impl Into<String>) -> Self {
        self.ai_prefix = ai_prefix.into();
        self
    }

    pub fn buffer(&self) -> String {
        self.buffer
            .read()
            .expect("conversation string buffer read lock poisoned")
            .clone()
    }

    pub fn try_save_context(
        &self,
        inputs: BTreeMap<String, Value>,
        outputs: BTreeMap<String, Value>,
    ) -> Result<(), LangChainError> {
        let (input, output) = resolve_input_output(
            &inputs,
            &outputs,
            self.input_key.as_deref(),
            self.output_key.as_deref(),
            &[self.memory_key.clone()],
        )?;

        let human_line = format!("{}: {input}", self.human_prefix);
        let ai_line = format!("{}: {output}", self.ai_prefix);
        let mut buffer = self
            .buffer
            .write()
            .expect("conversation string buffer write lock poisoned");

        if buffer.is_empty() {
            *buffer = format!("{human_line}\n{ai_line}");
        } else {
            buffer.push('\n');
            buffer.push_str(&human_line);
            buffer.push('\n');
            buffer.push_str(&ai_line);
        }

        Ok(())
    }
}

impl BaseMemory for ConversationStringBufferMemory {
    fn memory_variables(&self) -> Vec<String> {
        vec![self.memory_key.clone()]
    }

    fn load_memory_variables(&self, _inputs: BTreeMap<String, Value>) -> BTreeMap<String, Value> {
        [(self.memory_key.clone(), json!(self.buffer()))].into()
    }

    fn save_context(&self, inputs: BTreeMap<String, Value>, outputs: BTreeMap<String, Value>) {
        self.try_save_context(inputs, outputs)
            .expect("conversation string buffer memory should resolve input and output keys")
    }

    fn clear(&self) {
        self.buffer
            .write()
            .expect("conversation string buffer write lock poisoned")
            .clear();
    }
}

pub fn get_prompt_input_key(
    inputs: &BTreeMap<String, Value>,
    memory_variables: &[String],
) -> Result<String, LangChainError> {
    let ignored = memory_variables
        .iter()
        .cloned()
        .chain(std::iter::once("stop".to_owned()))
        .collect::<BTreeSet<_>>();
    let prompt_input_keys = inputs
        .keys()
        .filter(|key| !ignored.contains(*key))
        .cloned()
        .collect::<Vec<_>>();

    if prompt_input_keys.len() != 1 {
        return Err(LangChainError::request(format!(
            "one input key expected, got {:?}",
            prompt_input_keys
        )));
    }

    Ok(prompt_input_keys[0].clone())
}

fn resolve_input_output(
    inputs: &BTreeMap<String, Value>,
    outputs: &BTreeMap<String, Value>,
    input_key: Option<&str>,
    output_key: Option<&str>,
    memory_variables: &[String],
) -> Result<(String, String), LangChainError> {
    let input_key = match input_key {
        Some(input_key) => input_key.to_owned(),
        None => get_prompt_input_key(inputs, memory_variables)?,
    };
    let output_key = match output_key {
        Some(output_key) => output_key.to_owned(),
        None if outputs.len() == 1 => outputs
            .keys()
            .next()
            .expect("single output map should have one key")
            .clone(),
        None if outputs.contains_key(DEFAULT_OUTPUT_KEY) => DEFAULT_OUTPUT_KEY.to_owned(),
        None => {
            return Err(LangChainError::request(format!(
                "multiple output keys present; set output_key explicitly: {:?}",
                outputs.keys().collect::<Vec<_>>()
            )));
        }
    };

    let input = inputs.get(&input_key).ok_or_else(|| {
        LangChainError::request(format!("missing input key `{input_key}` for memory save"))
    })?;
    let output = outputs.get(&output_key).ok_or_else(|| {
        LangChainError::request(format!("missing output key `{output_key}` for memory save"))
    })?;

    Ok((value_to_string(input), value_to_string(output)))
}

fn value_to_string(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}
