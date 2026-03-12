use std::collections::BTreeMap;
use std::ops::Add;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::messages::{AIMessage, ResponseMetadata, UsageMetadata};

pub type GenerationInfo = BTreeMap<String, Value>;

fn merge_generation_info(
    left: Option<&GenerationInfo>,
    right: Option<&GenerationInfo>,
) -> Option<GenerationInfo> {
    let mut merged = left.cloned().unwrap_or_default();
    if let Some(right) = right {
        merged.extend(right.clone());
    }

    (!merged.is_empty()).then_some(merged)
}

fn merge_response_metadata(left: &ResponseMetadata, right: &ResponseMetadata) -> ResponseMetadata {
    let mut merged = left.clone();
    merged.extend(right.clone());
    merged
}

fn merge_usage_metadata(
    left: Option<&UsageMetadata>,
    right: Option<&UsageMetadata>,
) -> Option<UsageMetadata> {
    match (left, right) {
        (Some(left), Some(right)) => Some(UsageMetadata {
            input_tokens: left.input_tokens + right.input_tokens,
            output_tokens: left.output_tokens + right.output_tokens,
            total_tokens: left.total_tokens + right.total_tokens,
        }),
        (Some(left), None) => Some(left.clone()),
        (None, Some(right)) => Some(right.clone()),
        (None, None) => None,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Generation {
    text: String,
    generation_info: Option<GenerationInfo>,
}

impl Generation {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            generation_info: None,
        }
    }

    pub fn with_info(text: impl Into<String>, generation_info: GenerationInfo) -> Self {
        Self {
            text: text.into(),
            generation_info: Some(generation_info),
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn generation_info(&self) -> Option<&GenerationInfo> {
        self.generation_info.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationChunk {
    text: String,
    generation_info: Option<GenerationInfo>,
}

impl GenerationChunk {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            generation_info: None,
        }
    }

    pub fn with_info(text: impl Into<String>, generation_info: GenerationInfo) -> Self {
        Self {
            text: text.into(),
            generation_info: Some(generation_info),
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn generation_info(&self) -> Option<&GenerationInfo> {
        self.generation_info.as_ref()
    }
}

impl Add for GenerationChunk {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            text: format!("{}{}", self.text, rhs.text),
            generation_info: merge_generation_info(
                self.generation_info.as_ref(),
                rhs.generation_info.as_ref(),
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatGeneration {
    message: AIMessage,
    text: String,
    generation_info: Option<GenerationInfo>,
}

impl ChatGeneration {
    pub fn new(message: AIMessage) -> Self {
        Self {
            text: message.content().to_owned(),
            message,
            generation_info: None,
        }
    }

    pub fn with_info(message: AIMessage, generation_info: GenerationInfo) -> Self {
        Self {
            text: message.content().to_owned(),
            message,
            generation_info: Some(generation_info),
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn message(&self) -> &AIMessage {
        &self.message
    }

    pub fn generation_info(&self) -> Option<&GenerationInfo> {
        self.generation_info.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatGenerationChunk {
    message: AIMessage,
    text: String,
    generation_info: Option<GenerationInfo>,
}

impl ChatGenerationChunk {
    pub fn new(message: AIMessage) -> Self {
        Self {
            text: message.content().to_owned(),
            message,
            generation_info: None,
        }
    }

    pub fn with_info(message: AIMessage, generation_info: GenerationInfo) -> Self {
        Self {
            text: message.content().to_owned(),
            message,
            generation_info: Some(generation_info),
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn message(&self) -> &AIMessage {
        &self.message
    }

    pub fn generation_info(&self) -> Option<&GenerationInfo> {
        self.generation_info.as_ref()
    }
}

impl Add for ChatGenerationChunk {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let message = AIMessage::with_metadata(
            format!("{}{}", self.message.content(), rhs.message.content()),
            merge_response_metadata(
                self.message.response_metadata(),
                rhs.message.response_metadata(),
            ),
            merge_usage_metadata(self.message.usage_metadata(), rhs.message.usage_metadata()),
        );

        Self {
            text: message.content().to_owned(),
            message,
            generation_info: merge_generation_info(
                self.generation_info.as_ref(),
                rhs.generation_info.as_ref(),
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum GenerationCandidate {
    Text(Generation),
    Chat(ChatGeneration),
    TextChunk(GenerationChunk),
    ChatChunk(ChatGenerationChunk),
}

impl GenerationCandidate {
    pub fn text(&self) -> &str {
        match self {
            Self::Text(generation) => generation.text(),
            Self::Chat(generation) => generation.text(),
            Self::TextChunk(generation) => generation.text(),
            Self::ChatChunk(generation) => generation.text(),
        }
    }

    pub fn generation_info(&self) -> Option<&GenerationInfo> {
        match self {
            Self::Text(generation) => generation.generation_info(),
            Self::Chat(generation) => generation.generation_info(),
            Self::TextChunk(generation) => generation.generation_info(),
            Self::ChatChunk(generation) => generation.generation_info(),
        }
    }
}

impl From<Generation> for GenerationCandidate {
    fn from(value: Generation) -> Self {
        Self::Text(value)
    }
}

impl From<ChatGeneration> for GenerationCandidate {
    fn from(value: ChatGeneration) -> Self {
        Self::Chat(value)
    }
}

impl From<GenerationChunk> for GenerationCandidate {
    fn from(value: GenerationChunk) -> Self {
        Self::TextChunk(value)
    }
}

impl From<ChatGenerationChunk> for GenerationCandidate {
    fn from(value: ChatGenerationChunk) -> Self {
        Self::ChatChunk(value)
    }
}

pub type LLMGeneration = GenerationCandidate;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LLMResult {
    generations: Vec<Vec<GenerationCandidate>>,
    llm_output: Option<ResponseMetadata>,
}

impl LLMResult {
    pub fn new<G>(generations: Vec<Vec<G>>) -> Self
    where
        G: Into<GenerationCandidate>,
    {
        Self {
            generations: generations
                .into_iter()
                .map(|choices| choices.into_iter().map(Into::into).collect())
                .collect(),
            llm_output: None,
        }
    }

    pub fn with_output(mut self, llm_output: ResponseMetadata) -> Self {
        self.llm_output = Some(llm_output);
        self
    }

    pub fn generations(&self) -> &[Vec<GenerationCandidate>] {
        &self.generations
    }

    pub fn llm_output(&self) -> Option<&ResponseMetadata> {
        self.llm_output.as_ref()
    }

    pub fn primary_generation(&self) -> Option<&GenerationCandidate> {
        self.generations.first().and_then(|choices| choices.first())
    }

    pub fn primary_text(&self) -> Option<&str> {
        self.primary_generation().map(GenerationCandidate::text)
    }

    pub fn flatten(&self) -> Vec<Self> {
        self.generations
            .iter()
            .enumerate()
            .map(|(index, generations)| {
                // Mirror Python LangChain: token usage is only retained on the first
                // flattened item so callback accounting does not double count usage.
                let llm_output = self.llm_output.as_ref().map(|output| {
                    let mut pruned = output.clone();
                    if index > 0 && pruned.contains_key("token_usage") {
                        pruned.insert("token_usage".to_owned(), Value::Object(Default::default()));
                    }
                    pruned
                });

                Self {
                    generations: vec![generations.clone()],
                    llm_output,
                }
            })
            .collect()
    }
}
