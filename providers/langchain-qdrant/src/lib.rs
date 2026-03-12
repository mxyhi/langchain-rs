use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::documents::Document;
use langchain_core::embeddings::Embeddings;
use langchain_core::vectorstores::{InMemoryVectorStore, VectorStore};

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
    inner: InMemoryVectorStore<E>,
    retrieval_mode: RetrievalMode,
}

impl<E> QdrantVectorStore<E>
where
    E: Embeddings,
{
    pub fn new(embedding: E) -> Self {
        Self {
            inner: InMemoryVectorStore::new(embedding),
            retrieval_mode: RetrievalMode::Dense,
        }
    }

    pub fn with_retrieval_mode(mut self, retrieval_mode: RetrievalMode) -> Self {
        self.retrieval_mode = retrieval_mode;
        self
    }

    pub fn retrieval_mode(&self) -> RetrievalMode {
        self.retrieval_mode
    }

    pub fn embedding(&self) -> &E {
        self.inner.embedding()
    }
}

impl<E> VectorStore for QdrantVectorStore<E>
where
    E: Embeddings + 'static,
{
    fn add_documents<'a>(
        &'a mut self,
        documents: Vec<Document>,
    ) -> BoxFuture<'a, Result<Vec<String>, LangChainError>> {
        self.inner.add_documents(documents)
    }

    fn similarity_search<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        self.inner.similarity_search(query, limit)
    }

    fn get_by_ids(&self, ids: &[String]) -> Result<Vec<Document>, LangChainError> {
        self.inner.get_by_ids(ids)
    }

    fn delete(&mut self, ids: &[String]) -> Result<bool, LangChainError> {
        self.inner.delete(ids)
    }
}

pub type Qdrant<E> = QdrantVectorStore<E>;
