use langchain_core::LangChainError;
use langchain_core::documents::Document;
use langchain_core::embeddings::Embeddings;
use langchain_core::vectorstores::{InMemoryVectorStore, VectorStore};

pub struct Chroma<E> {
    collection_name: String,
    inner: InMemoryVectorStore<E>,
}

impl<E> Chroma<E>
where
    E: Embeddings,
{
    pub fn new(collection_name: impl Into<String>, embedding: E) -> Self {
        Self {
            collection_name: collection_name.into(),
            inner: InMemoryVectorStore::new(embedding),
        }
    }

    pub fn collection_name(&self) -> &str {
        &self.collection_name
    }

    pub fn embedding(&self) -> &E {
        self.inner.embedding()
    }
}

impl<E> VectorStore for Chroma<E>
where
    E: Embeddings + 'static,
{
    fn add_documents<'a>(
        &'a mut self,
        documents: Vec<Document>,
    ) -> futures_util::future::BoxFuture<'a, Result<Vec<String>, LangChainError>> {
        self.inner.add_documents(documents)
    }

    fn similarity_search<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> futures_util::future::BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        self.inner.similarity_search(query, limit)
    }

    fn get_by_ids(&self, ids: &[String]) -> Result<Vec<Document>, LangChainError> {
        self.inner.get_by_ids(ids)
    }

    fn delete(&mut self, ids: &[String]) -> Result<bool, LangChainError> {
        self.inner.delete(ids)
    }
}
