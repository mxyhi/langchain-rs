use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use futures_util::future::BoxFuture;
use serde_json::Value;

use crate::LangChainError;
use crate::documents::Document;
use crate::embeddings::Embeddings;
pub use crate::retrievers::VectorStoreRetriever;

pub trait VectorStore: Send + Sync {
    fn add_documents<'a>(
        &'a mut self,
        documents: Vec<Document>,
    ) -> BoxFuture<'a, Result<Vec<String>, LangChainError>>;

    fn similarity_search<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>>;

    fn get_by_ids(&self, _ids: &[String]) -> Result<Vec<Document>, LangChainError> {
        Err(LangChainError::unsupported(format!(
            "{} does not yet support get_by_ids",
            std::any::type_name::<Self>()
                .rsplit("::")
                .next()
                .unwrap_or("VectorStore"),
        )))
    }

    fn delete(&mut self, ids: &[String]) -> Result<bool, LangChainError> {
        let _ = ids;
        Err(LangChainError::unsupported(format!(
            "{} does not yet support delete",
            std::any::type_name::<Self>()
                .rsplit("::")
                .next()
                .unwrap_or("VectorStore"),
        )))
    }
}

#[derive(Debug, Clone)]
struct StoredDocument {
    id: String,
    document: Document,
    embedding: Vec<f32>,
}

pub struct InMemoryVectorStore<E> {
    embedding: Arc<E>,
    entries: Vec<StoredDocument>,
    next_id: usize,
}

impl<E> InMemoryVectorStore<E>
where
    E: Embeddings,
{
    pub fn new(embedding: E) -> Self {
        Self {
            embedding: Arc::new(embedding),
            entries: Vec::new(),
            next_id: 0,
        }
    }

    pub fn embedding(&self) -> &E {
        &self.embedding
    }
}

impl<E> VectorStore for InMemoryVectorStore<E>
where
    E: Embeddings + 'static,
{
    fn add_documents<'a>(
        &'a mut self,
        documents: Vec<Document>,
    ) -> BoxFuture<'a, Result<Vec<String>, LangChainError>> {
        Box::pin(async move {
            let texts = documents
                .iter()
                .map(|document| document.page_content.clone())
                .collect::<Vec<_>>();
            let embeddings = self.embedding.embed_documents(texts).await?;
            let mut ids = Vec::with_capacity(documents.len());

            for (mut document, embedding) in documents.into_iter().zip(embeddings.into_iter()) {
                let id = document.id.take().unwrap_or_else(|| {
                    let generated = format!("doc_{}", self.next_id);
                    self.next_id += 1;
                    generated
                });
                document.id = Some(id.clone());
                ids.push(id.clone());
                self.entries.push(StoredDocument {
                    id,
                    document,
                    embedding,
                });
            }

            Ok(ids)
        })
    }

    fn similarity_search<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        Box::pin(async move {
            let query_embedding = self.embedding.embed_query(query).await?;
            let mut scored = self
                .entries
                .iter()
                .map(|entry| {
                    // Dot product is enough because CharacterEmbeddings returns normalized
                    // vectors. Keeping the score local avoids extra allocations.
                    let score = query_embedding
                        .iter()
                        .zip(entry.embedding.iter())
                        .map(|(left, right)| left * right)
                        .sum::<f32>();
                    (score, entry.document.clone())
                })
                .collect::<Vec<_>>();

            scored.sort_by(|left, right| right.0.total_cmp(&left.0));
            Ok(scored
                .into_iter()
                .take(limit)
                .map(|(_, document)| document)
                .collect())
        })
    }

    fn get_by_ids(&self, ids: &[String]) -> Result<Vec<Document>, LangChainError> {
        let wanted = ids.iter().cloned().collect::<HashSet<_>>();
        Ok(self
            .entries
            .iter()
            .filter(|entry| wanted.contains(&entry.id))
            .map(|entry| entry.document.clone())
            .collect())
    }

    fn delete(&mut self, ids: &[String]) -> Result<bool, LangChainError> {
        let wanted = ids.iter().cloned().collect::<HashSet<_>>();
        let before = self.entries.len();
        self.entries.retain(|entry| !wanted.contains(&entry.id));
        Ok(before != self.entries.len())
    }
}

pub fn metadata(key: impl Into<String>, value: Value) -> BTreeMap<String, Value> {
    BTreeMap::from([(key.into(), value)])
}
