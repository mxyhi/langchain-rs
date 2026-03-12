use futures_util::future::BoxFuture;

use crate::LangChainError;
use crate::embeddings::Embeddings;
use crate::prompts::{PromptArgument, PromptArguments};

pub trait BaseExampleSelector: Send + Sync {
    fn select_examples<'a>(
        &'a self,
        input: &'a PromptArguments,
    ) -> BoxFuture<'a, Result<Vec<PromptArguments>, LangChainError>>;
}

#[derive(Debug, Clone, Default)]
pub struct LengthBasedExampleSelector {
    examples: Vec<PromptArguments>,
    max_length: usize,
}

impl LengthBasedExampleSelector {
    pub fn new(max_length: usize) -> Self {
        Self {
            examples: Vec::new(),
            max_length,
        }
    }

    pub fn with_examples(mut self, examples: Vec<PromptArguments>) -> Self {
        self.examples = examples;
        self
    }

    pub fn add_example(&mut self, example: PromptArguments) {
        self.examples.push(example);
    }

    pub fn examples(&self) -> &[PromptArguments] {
        &self.examples
    }
}

impl BaseExampleSelector for LengthBasedExampleSelector {
    fn select_examples<'a>(
        &'a self,
        input: &'a PromptArguments,
    ) -> BoxFuture<'a, Result<Vec<PromptArguments>, LangChainError>> {
        Box::pin(async move {
            let mut remaining = self
                .max_length
                .saturating_sub(serialized_arguments(input).len());
            let mut selected = Vec::new();

            for example in &self.examples {
                let length = serialized_arguments(example).len();
                if length > remaining {
                    break;
                }
                remaining = remaining.saturating_sub(length);
                selected.push(example.clone());
            }

            Ok(selected)
        })
    }
}

#[derive(Debug, Clone)]
struct EmbeddedExample {
    arguments: PromptArguments,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SemanticSimilarityExampleSelector<E> {
    embedding: E,
    k: usize,
    examples: Vec<EmbeddedExample>,
}

impl<E> SemanticSimilarityExampleSelector<E>
where
    E: Embeddings,
{
    pub fn new(embedding: E, k: usize) -> Self {
        Self {
            embedding,
            k: k.max(1),
            examples: Vec::new(),
        }
    }

    pub async fn add_example(&mut self, arguments: PromptArguments) -> Result<(), LangChainError> {
        let vector = self
            .embedding
            .embed_query(&serialized_arguments(&arguments))
            .await?;
        self.examples.push(EmbeddedExample {
            arguments,
            embedding: vector,
        });
        Ok(())
    }
}

impl<E> BaseExampleSelector for SemanticSimilarityExampleSelector<E>
where
    E: Embeddings + Send + Sync,
{
    fn select_examples<'a>(
        &'a self,
        input: &'a PromptArguments,
    ) -> BoxFuture<'a, Result<Vec<PromptArguments>, LangChainError>> {
        Box::pin(async move {
            let query_embedding = self
                .embedding
                .embed_query(&serialized_arguments(input))
                .await?;

            let mut ranked = self
                .examples
                .iter()
                .map(|example| {
                    let score = query_embedding
                        .iter()
                        .zip(example.embedding.iter())
                        .map(|(left, right)| left * right)
                        .sum::<f32>();
                    (score, example.arguments.clone())
                })
                .collect::<Vec<_>>();

            ranked.sort_by(|left, right| right.0.total_cmp(&left.0));
            Ok(ranked
                .into_iter()
                .take(self.k)
                .map(|(_, arguments)| arguments)
                .collect())
        })
    }
}

#[derive(Debug, Clone)]
pub struct MaxMarginalRelevanceExampleSelector<E> {
    inner: SemanticSimilarityExampleSelector<E>,
}

impl<E> MaxMarginalRelevanceExampleSelector<E>
where
    E: Embeddings,
{
    pub fn new(embedding: E, k: usize) -> Self {
        Self {
            inner: SemanticSimilarityExampleSelector::new(embedding, k),
        }
    }

    pub async fn add_example(&mut self, arguments: PromptArguments) -> Result<(), LangChainError> {
        self.inner.add_example(arguments).await
    }
}

impl<E> BaseExampleSelector for MaxMarginalRelevanceExampleSelector<E>
where
    E: Embeddings + Send + Sync,
{
    fn select_examples<'a>(
        &'a self,
        input: &'a PromptArguments,
    ) -> BoxFuture<'a, Result<Vec<PromptArguments>, LangChainError>> {
        self.inner.select_examples(input)
    }
}

pub fn sorted_values(arguments: &PromptArguments) -> Vec<String> {
    let mut entries = arguments.iter().collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(right.0));
    entries
        .into_iter()
        .map(|(_, value)| serialize_argument_value(value))
        .collect()
}

fn serialized_arguments(arguments: &PromptArguments) -> String {
    let mut entries = arguments.iter().collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(right.0));

    entries
        .into_iter()
        .map(|(name, value)| format!("{name}={}", serialize_argument_value(value)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn serialize_argument_value(value: &PromptArgument) -> String {
    match value {
        PromptArgument::String(text) => text.clone(),
        PromptArgument::Messages(messages) => messages
            .iter()
            .map(|message| format!("{}:{}", message.role().as_str(), message.content()))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}
