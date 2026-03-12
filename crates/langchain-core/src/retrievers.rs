use futures_util::future::BoxFuture;

use crate::LangChainError;
use crate::documents::Document;
use crate::runnables::{Runnable, RunnableConfig};
use crate::vectorstores::VectorStore;

pub trait BaseRetriever: Send + Sync {
    fn get_relevant_documents<'a>(
        &'a self,
        query: &'a str,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>>;
}

impl<T> Runnable<String, Vec<Document>> for T
where
    T: BaseRetriever,
{
    fn invoke<'a>(
        &'a self,
        input: String,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        Box::pin(async move { self.get_relevant_documents(&input, config).await })
    }
}

pub struct VectorStoreRetriever<V> {
    store: V,
    limit: usize,
}

impl<V> VectorStoreRetriever<V> {
    pub fn new(store: V) -> Self {
        Self { store, limit: 4 }
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit.max(1);
        self
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    pub fn store(&self) -> &V {
        &self.store
    }
}

impl<V> BaseRetriever for VectorStoreRetriever<V>
where
    V: VectorStore + Send + Sync + 'static,
{
    fn get_relevant_documents<'a>(
        &'a self,
        query: &'a str,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        self.store.similarity_search(query, self.limit)
    }
}
