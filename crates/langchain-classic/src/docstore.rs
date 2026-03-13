use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use langchain_core::LangChainError;
use langchain_core::documents::Document;

#[derive(Debug, Clone, Default)]
pub struct InMemoryDocstore {
    entries: Arc<RwLock<HashMap<String, Document>>>,
}

impl InMemoryDocstore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_document(self, key: impl Into<String>, document: Document) -> Self {
        self.add([(key.into(), document)]);
        self
    }

    pub fn add<I>(&self, documents: I)
    where
        I: IntoIterator<Item = (String, Document)>,
    {
        let mut entries = self.entries.write().expect("docstore write lock poisoned");
        for (key, document) in documents {
            entries.insert(key, document);
        }
    }

    pub fn search(&self, key: &str) -> Option<Document> {
        self.entries
            .read()
            .expect("docstore read lock poisoned")
            .get(key)
            .cloned()
    }
}

pub type DocstoreFn = Box<dyn Fn(&str) -> Option<Document> + Send + Sync>;

#[derive(Debug, Clone, Copy, Default)]
pub struct Wikipedia;

impl Wikipedia {
    pub fn new() -> Self {
        Self
    }

    pub fn search(&self, _query: &str) -> Result<Option<Document>, LangChainError> {
        Err(LangChainError::unsupported(
            "langchain-classic Wikipedia docstore is not implemented in this Rust workspace",
        ))
    }
}

pub use langchain_core::documents::Document as DocstoreDocument;
