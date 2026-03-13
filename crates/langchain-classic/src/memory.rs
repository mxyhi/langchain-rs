use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::RwLock;

use langchain_core::messages::{
    AIMessage, BaseMessage, HumanMessage, MessageRole, messages_to_dict,
};
use serde_json::{Value, json};

use crate::LangChainError;
use crate::base_memory::BaseMemory;

pub mod prompt;

pub use langchain_core::chat_history::{BaseChatMessageHistory, InMemoryChatMessageHistory};

const DEFAULT_MEMORY_KEY: &str = "history";
const DEFAULT_OUTPUT_KEY: &str = "output";

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryBuffer {
    Text(String),
    Messages(Vec<BaseMessage>),
}

impl MemoryBuffer {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Messages(_) => None,
        }
    }

    pub fn as_messages(&self) -> Option<&[BaseMessage]> {
        match self {
            Self::Text(_) => None,
            Self::Messages(messages) => Some(messages),
        }
    }

    pub fn into_messages(self) -> Option<Vec<BaseMessage>> {
        match self {
            Self::Text(_) => None,
            Self::Messages(messages) => Some(messages),
        }
    }

    fn into_json(self) -> Value {
        match self {
            Self::Text(text) => json!(text),
            Self::Messages(messages) => json!(messages_to_dict(&messages)),
        }
    }
}

#[derive(Debug, Default)]
pub struct SimpleMemory {
    memories: BTreeMap<String, Value>,
}

impl SimpleMemory {
    pub fn new(memories: impl IntoIterator<Item = (impl Into<String>, Value)>) -> Self {
        Self {
            memories: memories
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect(),
        }
    }
}

impl BaseMemory for SimpleMemory {
    fn memory_variables(&self) -> Vec<String> {
        self.memories.keys().cloned().collect()
    }

    fn load_memory_variables(&self, _inputs: BTreeMap<String, Value>) -> BTreeMap<String, Value> {
        self.memories.clone()
    }

    fn save_context(&self, _inputs: BTreeMap<String, Value>, _outputs: BTreeMap<String, Value>) {}

    fn clear(&self) {}
}

#[derive(Debug)]
pub struct ReadOnlySharedMemory<M> {
    memory: M,
}

impl<M> ReadOnlySharedMemory<M> {
    pub fn new(memory: M) -> Self {
        Self { memory }
    }

    pub fn inner(&self) -> &M {
        &self.memory
    }
}

impl<M> BaseMemory for ReadOnlySharedMemory<M>
where
    M: BaseMemory,
{
    fn memory_variables(&self) -> Vec<String> {
        self.memory.memory_variables()
    }

    fn load_memory_variables(&self, inputs: BTreeMap<String, Value>) -> BTreeMap<String, Value> {
        self.memory.load_memory_variables(inputs)
    }

    fn save_context(&self, _inputs: BTreeMap<String, Value>, _outputs: BTreeMap<String, Value>) {}

    fn clear(&self) {}
}

pub struct ConversationBufferMemory<H = InMemoryChatMessageHistory> {
    chat_memory: H,
    input_key: Option<String>,
    output_key: Option<String>,
    return_messages: bool,
    memory_key: String,
    human_prefix: String,
    ai_prefix: String,
}

impl<H> fmt::Debug for ConversationBufferMemory<H> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConversationBufferMemory")
            .field("input_key", &self.input_key)
            .field("output_key", &self.output_key)
            .field("return_messages", &self.return_messages)
            .field("memory_key", &self.memory_key)
            .field("human_prefix", &self.human_prefix)
            .field("ai_prefix", &self.ai_prefix)
            .finish()
    }
}

impl Default for ConversationBufferMemory<InMemoryChatMessageHistory> {
    fn default() -> Self {
        Self {
            chat_memory: InMemoryChatMessageHistory::new(),
            input_key: None,
            output_key: None,
            return_messages: false,
            memory_key: DEFAULT_MEMORY_KEY.to_owned(),
            human_prefix: "Human".to_owned(),
            ai_prefix: "AI".to_owned(),
        }
    }
}

impl ConversationBufferMemory<InMemoryChatMessageHistory> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<H> ConversationBufferMemory<H> {
    pub fn with_chat_memory<T>(self, chat_memory: T) -> ConversationBufferMemory<T> {
        ConversationBufferMemory {
            chat_memory,
            input_key: self.input_key,
            output_key: self.output_key,
            return_messages: self.return_messages,
            memory_key: self.memory_key,
            human_prefix: self.human_prefix,
            ai_prefix: self.ai_prefix,
        }
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

    pub fn with_human_prefix(mut self, human_prefix: impl Into<String>) -> Self {
        self.human_prefix = human_prefix.into();
        self
    }

    pub fn with_ai_prefix(mut self, ai_prefix: impl Into<String>) -> Self {
        self.ai_prefix = ai_prefix.into();
        self
    }
}

impl<H> ConversationBufferMemory<H>
where
    H: BaseChatMessageHistory,
{
    pub fn buffer_as_messages(&self) -> Vec<BaseMessage> {
        self.chat_memory.messages()
    }

    pub fn buffer_as_str(&self) -> Result<String, LangChainError> {
        render_buffer_string(
            &self.buffer_as_messages(),
            &self.human_prefix,
            &self.ai_prefix,
        )
    }

    pub async fn abuffer_as_messages(&self) -> Vec<BaseMessage> {
        self.chat_memory.aget_messages().await
    }

    pub async fn abuffer_as_str(&self) -> Result<String, LangChainError> {
        let messages = self.chat_memory.aget_messages().await;
        render_buffer_string(&messages, &self.human_prefix, &self.ai_prefix)
    }

    pub fn buffer(&self) -> MemoryBuffer {
        if self.return_messages {
            MemoryBuffer::Messages(self.buffer_as_messages())
        } else {
            MemoryBuffer::Text(
                self.buffer_as_str()
                    .expect("conversation buffer memory should render to a string"),
            )
        }
    }

    pub async fn abuffer(&self) -> MemoryBuffer {
        if self.return_messages {
            MemoryBuffer::Messages(self.abuffer_as_messages().await)
        } else {
            MemoryBuffer::Text(
                self.abuffer_as_str()
                    .await
                    .expect("conversation buffer memory should render to a string"),
            )
        }
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

    pub async fn try_asave_context(
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

        self.chat_memory
            .aadd_messages(vec![
                HumanMessage::new(input).into(),
                AIMessage::new(output).into(),
            ])
            .await;
        Ok(())
    }
}

impl<H> BaseMemory for ConversationBufferMemory<H>
where
    H: BaseChatMessageHistory,
{
    fn memory_variables(&self) -> Vec<String> {
        vec![self.memory_key.clone()]
    }

    fn load_memory_variables(&self, _inputs: BTreeMap<String, Value>) -> BTreeMap<String, Value> {
        [(self.memory_key.clone(), self.buffer().into_json())].into()
    }

    fn save_context(&self, inputs: BTreeMap<String, Value>, outputs: BTreeMap<String, Value>) {
        self.try_save_context(inputs, outputs)
            .expect("conversation buffer memory should resolve input and output keys")
    }

    fn clear(&self) {
        self.chat_memory.clear();
    }

    fn aload_memory_variables<'a>(
        &'a self,
        _inputs: BTreeMap<String, Value>,
    ) -> futures_util::future::BoxFuture<'a, BTreeMap<String, Value>> {
        Box::pin(
            async move { [(self.memory_key.clone(), self.abuffer().await.into_json())].into() },
        )
    }

    fn asave_context<'a>(
        &'a self,
        inputs: BTreeMap<String, Value>,
        outputs: BTreeMap<String, Value>,
    ) -> futures_util::future::BoxFuture<'a, ()> {
        Box::pin(async move {
            self.try_asave_context(inputs, outputs)
                .await
                .expect("conversation buffer memory should resolve input and output keys");
        })
    }

    fn aclear<'a>(&'a self) -> futures_util::future::BoxFuture<'a, ()> {
        Box::pin(async move {
            self.chat_memory.aclear().await;
        })
    }
}

pub struct ConversationBufferWindowMemory<H = InMemoryChatMessageHistory> {
    chat_memory: H,
    input_key: Option<String>,
    output_key: Option<String>,
    return_messages: bool,
    memory_key: String,
    human_prefix: String,
    ai_prefix: String,
    k: usize,
}

impl<H> fmt::Debug for ConversationBufferWindowMemory<H> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConversationBufferWindowMemory")
            .field("input_key", &self.input_key)
            .field("output_key", &self.output_key)
            .field("return_messages", &self.return_messages)
            .field("memory_key", &self.memory_key)
            .field("human_prefix", &self.human_prefix)
            .field("ai_prefix", &self.ai_prefix)
            .field("k", &self.k)
            .finish()
    }
}

impl Default for ConversationBufferWindowMemory<InMemoryChatMessageHistory> {
    fn default() -> Self {
        Self {
            chat_memory: InMemoryChatMessageHistory::new(),
            input_key: None,
            output_key: None,
            return_messages: false,
            memory_key: DEFAULT_MEMORY_KEY.to_owned(),
            human_prefix: "Human".to_owned(),
            ai_prefix: "AI".to_owned(),
            k: 5,
        }
    }
}

impl ConversationBufferWindowMemory<InMemoryChatMessageHistory> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<H> ConversationBufferWindowMemory<H> {
    pub fn with_chat_memory<T>(self, chat_memory: T) -> ConversationBufferWindowMemory<T> {
        ConversationBufferWindowMemory {
            chat_memory,
            input_key: self.input_key,
            output_key: self.output_key,
            return_messages: self.return_messages,
            memory_key: self.memory_key,
            human_prefix: self.human_prefix,
            ai_prefix: self.ai_prefix,
            k: self.k,
        }
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

    pub fn with_human_prefix(mut self, human_prefix: impl Into<String>) -> Self {
        self.human_prefix = human_prefix.into();
        self
    }

    pub fn with_ai_prefix(mut self, ai_prefix: impl Into<String>) -> Self {
        self.ai_prefix = ai_prefix.into();
        self
    }

    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }
}

impl<H> ConversationBufferWindowMemory<H>
where
    H: BaseChatMessageHistory,
{
    pub fn buffer_as_messages(&self) -> Vec<BaseMessage> {
        trim_window_messages(self.chat_memory.messages(), self.k)
    }

    pub fn buffer_as_str(&self) -> Result<String, LangChainError> {
        render_buffer_string(
            &self.buffer_as_messages(),
            &self.human_prefix,
            &self.ai_prefix,
        )
    }

    pub async fn abuffer_as_messages(&self) -> Vec<BaseMessage> {
        trim_window_messages(self.chat_memory.aget_messages().await, self.k)
    }

    pub async fn abuffer_as_str(&self) -> Result<String, LangChainError> {
        let messages = self.abuffer_as_messages().await;
        render_buffer_string(&messages, &self.human_prefix, &self.ai_prefix)
    }

    pub fn buffer(&self) -> MemoryBuffer {
        if self.return_messages {
            MemoryBuffer::Messages(self.buffer_as_messages())
        } else {
            MemoryBuffer::Text(
                self.buffer_as_str()
                    .expect("conversation buffer window memory should render to a string"),
            )
        }
    }

    pub async fn abuffer(&self) -> MemoryBuffer {
        if self.return_messages {
            MemoryBuffer::Messages(self.abuffer_as_messages().await)
        } else {
            MemoryBuffer::Text(
                self.abuffer_as_str()
                    .await
                    .expect("conversation buffer window memory should render to a string"),
            )
        }
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

    pub async fn try_asave_context(
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

        self.chat_memory
            .aadd_messages(vec![
                HumanMessage::new(input).into(),
                AIMessage::new(output).into(),
            ])
            .await;
        Ok(())
    }
}

impl<H> BaseMemory for ConversationBufferWindowMemory<H>
where
    H: BaseChatMessageHistory,
{
    fn memory_variables(&self) -> Vec<String> {
        vec![self.memory_key.clone()]
    }

    fn load_memory_variables(&self, _inputs: BTreeMap<String, Value>) -> BTreeMap<String, Value> {
        [(self.memory_key.clone(), self.buffer().into_json())].into()
    }

    fn save_context(&self, inputs: BTreeMap<String, Value>, outputs: BTreeMap<String, Value>) {
        self.try_save_context(inputs, outputs)
            .expect("conversation buffer window memory should resolve input and output keys")
    }

    fn clear(&self) {
        self.chat_memory.clear();
    }

    fn aload_memory_variables<'a>(
        &'a self,
        _inputs: BTreeMap<String, Value>,
    ) -> futures_util::future::BoxFuture<'a, BTreeMap<String, Value>> {
        Box::pin(
            async move { [(self.memory_key.clone(), self.abuffer().await.into_json())].into() },
        )
    }

    fn asave_context<'a>(
        &'a self,
        inputs: BTreeMap<String, Value>,
        outputs: BTreeMap<String, Value>,
    ) -> futures_util::future::BoxFuture<'a, ()> {
        Box::pin(async move {
            self.try_asave_context(inputs, outputs)
                .await
                .expect("conversation buffer window memory should resolve input and output keys");
        })
    }

    fn aclear<'a>(&'a self) -> futures_util::future::BoxFuture<'a, ()> {
        Box::pin(async move {
            self.chat_memory.aclear().await;
        })
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

        buffer.push('\n');
        buffer.push_str(&human_line);
        buffer.push('\n');
        buffer.push_str(&ai_line);

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

pub struct CombinedMemory {
    memories: Vec<Box<dyn BaseMemory>>,
}

impl CombinedMemory {
    pub fn new(memories: Vec<Box<dyn BaseMemory>>) -> Result<Self, LangChainError> {
        let mut all_variables = BTreeSet::new();
        let mut duplicates = BTreeSet::new();

        for memory in &memories {
            for variable in memory.memory_variables() {
                if !all_variables.insert(variable.clone()) {
                    duplicates.insert(variable);
                }
            }
        }

        if !duplicates.is_empty() {
            return Err(LangChainError::request(format!(
                "The same variables {:?} are found in multiple memory objects, which is not allowed by CombinedMemory.",
                duplicates
            )));
        }

        Ok(Self { memories })
    }
}

impl fmt::Debug for CombinedMemory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CombinedMemory")
            .field("memory_variables", &self.memory_variables())
            .finish()
    }
}

impl BaseMemory for CombinedMemory {
    fn memory_variables(&self) -> Vec<String> {
        self.memories
            .iter()
            .flat_map(|memory| memory.memory_variables())
            .collect()
    }

    fn load_memory_variables(&self, inputs: BTreeMap<String, Value>) -> BTreeMap<String, Value> {
        let mut memory_data = BTreeMap::new();

        for memory in &self.memories {
            for (key, value) in memory.load_memory_variables(inputs.clone()) {
                if memory_data.insert(key.clone(), value).is_some() {
                    panic!("The variable {key} is repeated in the CombinedMemory.");
                }
            }
        }

        memory_data
    }

    fn save_context(&self, inputs: BTreeMap<String, Value>, outputs: BTreeMap<String, Value>) {
        for memory in &self.memories {
            memory.save_context(inputs.clone(), outputs.clone());
        }
    }

    fn clear(&self) {
        for memory in &self.memories {
            memory.clear();
        }
    }
}

fn render_buffer_string(
    messages: &[BaseMessage],
    human_prefix: &str,
    ai_prefix: &str,
) -> Result<String, LangChainError> {
    messages
        .iter()
        .map(|message| render_buffer_line(message, human_prefix, ai_prefix))
        .collect::<Result<Vec<_>, _>>()
        .map(|lines| lines.join("\n"))
}

fn render_buffer_line(
    message: &BaseMessage,
    human_prefix: &str,
    ai_prefix: &str,
) -> Result<String, LangChainError> {
    let label = match message.role() {
        MessageRole::Human => human_prefix,
        MessageRole::Ai => ai_prefix,
        MessageRole::System => "System",
        MessageRole::Tool => "Tool",
    };

    Ok(format!("{label}: {}", message.content()))
}

fn trim_window_messages(messages: Vec<BaseMessage>, k: usize) -> Vec<BaseMessage> {
    if k == 0 {
        return Vec::new();
    }

    let window_size = k.saturating_mul(2);
    let message_count = messages.len();

    if message_count <= window_size {
        return messages;
    }

    // `k` is counted in turns, so the visible window keeps the last 2*k messages.
    messages
        .into_iter()
        .skip(message_count - window_size)
        .collect()
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
            "One input key expected got {:?}",
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
                "Got multiple output keys: {:?}, cannot determine which to store in memory. Please set the 'output_key' explicitly.",
                outputs.keys().collect::<Vec<_>>(),
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
