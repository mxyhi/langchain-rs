use std::collections::BTreeMap;
use std::sync::Arc;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::documents::Document;
use langchain_core::embeddings::Embeddings;
use langchain_core::vectorstores::{InMemoryVectorStore, VectorStore};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::runtime::{Handle, Runtime};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMode {
    Dense,
    Sparse,
    Hybrid,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseVector {
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
}

pub trait SparseEmbeddings: Send + Sync {
    fn embed_query_sparse<'a>(
        &'a self,
        text: &'a str,
    ) -> BoxFuture<'a, Result<SparseVector, LangChainError>>;

    fn embed_documents_sparse<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<SparseVector>, LangChainError>>;
}

#[derive(Debug, Clone, Default)]
pub struct FastEmbedSparse;

impl FastEmbedSparse {
    pub fn new() -> Self {
        Self
    }

    fn encode(text: &str) -> SparseVector {
        let mut counts = [0.0_f32; 26];
        for byte in text.bytes() {
            let normalized = byte.to_ascii_lowercase();
            if normalized.is_ascii_lowercase() {
                counts[usize::from(normalized - b'a')] += 1.0;
            }
        }

        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (index, value) in counts.into_iter().enumerate() {
            if value > 0.0 {
                indices.push(index);
                values.push(value);
            }
        }

        SparseVector { indices, values }
    }
}

impl SparseEmbeddings for FastEmbedSparse {
    fn embed_query_sparse<'a>(
        &'a self,
        text: &'a str,
    ) -> BoxFuture<'a, Result<SparseVector, LangChainError>> {
        Box::pin(async move { Ok(Self::encode(text)) })
    }

    fn embed_documents_sparse<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<SparseVector>, LangChainError>> {
        Box::pin(async move { Ok(texts.into_iter().map(|text| Self::encode(&text)).collect()) })
    }
}

pub struct QdrantVectorStore<E> {
    backend: Backend<E>,
    retrieval_mode: RetrievalMode,
    next_id: usize,
}

enum Backend<E> {
    Local(InMemoryVectorStore<E>),
    Remote(RemoteQdrant<E>),
}

struct RemoteQdrant<E> {
    embedding: Arc<E>,
    client: Client,
    base_url: String,
    collection_name: String,
    api_key: Option<String>,
}

#[derive(Debug, Deserialize)]
struct QueryEnvelope {
    result: QueryResult,
}

#[derive(Debug, Deserialize)]
struct QueryResult {
    points: Vec<ScoredPoint>,
}

#[derive(Debug, Deserialize)]
struct ScoredPoint {
    id: Value,
    payload: PointPayload,
}

#[derive(Debug, Deserialize)]
struct PointEnvelope {
    result: Vec<PointRecord>,
}

#[derive(Debug, Deserialize)]
struct PointRecord {
    id: Value,
    payload: PointPayload,
}

#[derive(Debug, Deserialize)]
struct PointPayload {
    page_content: String,
    metadata: BTreeMap<String, Value>,
}

impl<E> QdrantVectorStore<E>
where
    E: Embeddings,
{
    pub fn new(embedding: E) -> Self {
        Self {
            backend: Backend::Local(InMemoryVectorStore::new(embedding)),
            retrieval_mode: RetrievalMode::Dense,
            next_id: 0,
        }
    }

    pub fn new_remote(
        collection_name: impl Into<String>,
        base_url: impl Into<String>,
        embedding: E,
    ) -> Self {
        Self {
            backend: Backend::Remote(RemoteQdrant {
                embedding: Arc::new(embedding),
                client: Client::new(),
                base_url: base_url.into().trim_end_matches('/').to_owned(),
                collection_name: collection_name.into(),
                api_key: None,
            }),
            retrieval_mode: RetrievalMode::Dense,
            next_id: 0,
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        if let Backend::Remote(remote) = &mut self.backend {
            remote.api_key = Some(api_key.into());
        }
        self
    }

    pub fn with_retrieval_mode(mut self, retrieval_mode: RetrievalMode) -> Self {
        self.retrieval_mode = retrieval_mode;
        self
    }

    pub fn retrieval_mode(&self) -> RetrievalMode {
        self.retrieval_mode
    }

    pub fn embedding(&self) -> &E {
        match &self.backend {
            Backend::Local(inner) => inner.embedding(),
            Backend::Remote(remote) => remote.embedding.as_ref(),
        }
    }

    async fn similarity_search_remote(
        &self,
        remote: &RemoteQdrant<E>,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Document>, LangChainError> {
        let query_vector = remote.embedding.embed_query(query).await?;
        let response = remote
            .request(
                "POST",
                format!("/collections/{}/points/query", remote.collection_name),
            )
            .json(&json!({
                "query": query_vector,
                "limit": limit,
                "with_payload": true,
            }))
            .send()
            .await
            .map_err(http_error("query qdrant points"))?
            .error_for_status()
            .map_err(http_error("query qdrant points"))?
            .json::<QueryEnvelope>()
            .await
            .map_err(http_error("decode qdrant query response"))?;

        Ok(response
            .result
            .points
            .into_iter()
            .map(|point| Document {
                page_content: point.payload.page_content,
                metadata: point.payload.metadata,
                id: Some(point_id_to_string(point.id)),
            })
            .collect())
    }

    async fn get_by_ids_remote(
        &self,
        remote: &RemoteQdrant<E>,
        ids: &[String],
    ) -> Result<Vec<Document>, LangChainError> {
        let response = remote
            .request(
                "POST",
                format!("/collections/{}/points", remote.collection_name),
            )
            .json(&json!({
                "ids": ids,
                "with_payload": true,
                "with_vector": false,
            }))
            .send()
            .await
            .map_err(http_error("get qdrant points"))?
            .error_for_status()
            .map_err(http_error("get qdrant points"))?
            .json::<PointEnvelope>()
            .await
            .map_err(http_error("decode qdrant get response"))?;

        Ok(response
            .result
            .into_iter()
            .map(|point| Document {
                page_content: point.payload.page_content,
                metadata: point.payload.metadata,
                id: Some(point_id_to_string(point.id)),
            })
            .collect())
    }

    async fn delete_remote(
        remote: &RemoteQdrant<E>,
        ids: &[String],
    ) -> Result<bool, LangChainError> {
        remote
            .request(
                "POST",
                format!("/collections/{}/points/delete", remote.collection_name),
            )
            .json(&json!({
                "points": ids,
            }))
            .send()
            .await
            .map_err(http_error("delete qdrant points"))?
            .error_for_status()
            .map_err(http_error("delete qdrant points"))?;
        Ok(true)
    }
}

impl<E> VectorStore for QdrantVectorStore<E>
where
    E: Embeddings + Send + Sync + 'static,
{
    fn add_documents<'a>(
        &'a mut self,
        documents: Vec<Document>,
    ) -> BoxFuture<'a, Result<Vec<String>, LangChainError>> {
        Box::pin(async move {
            match &mut self.backend {
                Backend::Local(inner) => inner.add_documents(documents).await,
                Backend::Remote(remote) => {
                    let ids = documents
                        .iter()
                        .map(|document| {
                            document.id.clone().unwrap_or_else(|| {
                                let generated = format!("doc_{}", self.next_id);
                                self.next_id += 1;
                                generated
                            })
                        })
                        .collect::<Vec<_>>();
                    let texts = documents
                        .iter()
                        .map(|document| document.page_content.clone())
                        .collect::<Vec<_>>();
                    let embeddings = remote.embedding.embed_documents(texts).await?;

                    let points = ids
                        .iter()
                        .zip(embeddings.into_iter())
                        .zip(documents.into_iter())
                        .map(|((id, vector), document)| {
                            json!({
                                "id": id,
                                "vector": vector,
                                "payload": {
                                    "page_content": document.page_content,
                                    "metadata": document.metadata,
                                }
                            })
                        })
                        .collect::<Vec<_>>();

                    remote
                        .request(
                            "PUT",
                            format!("/collections/{}/points", remote.collection_name),
                        )
                        .json(&json!({ "points": points }))
                        .send()
                        .await
                        .map_err(http_error("upsert qdrant points"))?
                        .error_for_status()
                        .map_err(http_error("upsert qdrant points"))?;

                    Ok(ids)
                }
            }
        })
    }

    fn similarity_search<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        Box::pin(async move {
            match &self.backend {
                Backend::Local(inner) => inner.similarity_search(query, limit).await,
                Backend::Remote(remote) => {
                    self.similarity_search_remote(remote, query, limit).await
                }
            }
        })
    }

    fn get_by_ids(&self, ids: &[String]) -> Result<Vec<Document>, LangChainError> {
        match &self.backend {
            Backend::Local(inner) => inner.get_by_ids(ids),
            Backend::Remote(remote) => block_on_sync(self.get_by_ids_remote(remote, ids)),
        }
    }

    fn delete(&mut self, ids: &[String]) -> Result<bool, LangChainError> {
        match &mut self.backend {
            Backend::Local(inner) => inner.delete(ids),
            Backend::Remote(remote) => block_on_sync(Self::delete_remote(remote, ids)),
        }
    }
}

impl<E> RemoteQdrant<E> {
    fn request(&self, method: &str, path: String) -> reqwest::RequestBuilder {
        let request = self.client.request(
            method.parse().expect("valid HTTP method"),
            format!("{}{}", self.base_url, path),
        );
        if let Some(api_key) = &self.api_key {
            request.header("api-key", api_key)
        } else {
            request
        }
    }
}

fn point_id_to_string(id: Value) -> String {
    match id {
        Value::String(value) => value,
        Value::Number(value) => value.to_string(),
        other => other.to_string(),
    }
}

fn http_error(context: &'static str) -> impl Fn(reqwest::Error) -> LangChainError {
    move |error| LangChainError::request(format!("{context}: {error}"))
}

fn block_on_sync<F, T>(future: F) -> Result<T, LangChainError>
where
    F: std::future::Future<Output = Result<T, LangChainError>>,
{
    match Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(future)),
        Err(_) => Runtime::new()
            .map_err(|error| LangChainError::request(format!("create runtime: {error}")))?
            .block_on(future),
    }
}

pub type Qdrant<E> = QdrantVectorStore<E>;

pub mod fastembed_sparse {
    pub use crate::FastEmbedSparse;
}

pub mod qdrant {
    pub use crate::Qdrant;
}

pub mod sparse_embeddings {
    pub use crate::{SparseEmbeddings, SparseVector};
}

pub mod vectorstores {
    pub use crate::{QdrantVectorStore, RetrievalMode};
}
