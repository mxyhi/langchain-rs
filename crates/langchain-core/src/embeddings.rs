use futures_util::future::BoxFuture;

use crate::LangChainError;

pub trait Embeddings: Send + Sync {
    fn embed_query<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>, LangChainError>>;

    fn embed_documents<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, LangChainError>>;
}

#[derive(Debug, Clone, Default)]
pub struct CharacterEmbeddings;

impl CharacterEmbeddings {
    pub fn new() -> Self {
        Self
    }

    fn embed(text: &str) -> Vec<f32> {
        // A tiny deterministic embedding for tests and in-memory usage.
        let mut vector = vec![0.0_f32; 26];
        for byte in text.bytes() {
            let normalized = byte.to_ascii_lowercase();
            if normalized.is_ascii_lowercase() {
                let index = usize::from(normalized - b'a');
                vector[index] += 1.0;
            }
        }

        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }

        vector
    }
}

impl Embeddings for CharacterEmbeddings {
    fn embed_query<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>, LangChainError>> {
        Box::pin(async move { Ok(Self::embed(text)) })
    }

    fn embed_documents<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, LangChainError>> {
        Box::pin(async move { Ok(texts.into_iter().map(|text| Self::embed(&text)).collect()) })
    }
}
