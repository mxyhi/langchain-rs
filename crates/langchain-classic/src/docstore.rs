use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};

use langchain_core::LangChainError;
use langchain_core::documents::Document;

#[derive(Debug)]
pub struct DocstoreSearchResult(Result<Option<Document>, LangChainError>);

impl DocstoreSearchResult {
    pub fn expect(self, message: &str) -> Option<Document> {
        self.0.expect(message)
    }

    pub fn ok(self) -> Option<Option<Document>> {
        self.0.ok()
    }
}

impl From<Option<Document>> for DocstoreSearchResult {
    fn from(value: Option<Document>) -> Self {
        Self(Ok(value))
    }
}

impl From<Result<Option<Document>, LangChainError>> for DocstoreSearchResult {
    fn from(value: Result<Option<Document>, LangChainError>) -> Self {
        Self(value)
    }
}

impl PartialEq<Option<Document>> for DocstoreSearchResult {
    fn eq(&self, other: &Option<Document>) -> bool {
        matches!(&self.0, Ok(value) if value == other)
    }
}

#[derive(Debug, Clone, Default)]
pub struct InMemoryDocstore {
    entries: Arc<RwLock<HashMap<String, Document>>>,
}

impl InMemoryDocstore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_document(self, key: impl Into<String>, document: Document) -> Self {
        self.add(key, document);
        self
    }

    pub fn add(&self, key: impl Into<String>, document: Document) {
        self.add_many([(key.into(), document)]);
    }

    pub fn add_many<I>(&self, documents: I)
    where
        I: IntoIterator<Item = (String, Document)>,
    {
        let mut entries = self.entries.write().expect("docstore write lock poisoned");
        for (key, document) in documents {
            entries.insert(key, document);
        }
    }

    pub fn search(&self, key: &str) -> DocstoreSearchResult {
        self.entries
            .read()
            .expect("docstore read lock poisoned")
            .get(key)
            .cloned()
            .into()
    }
}

#[derive(Clone)]
pub struct DocstoreFn {
    lookup: Arc<dyn Fn(&str) -> Option<Document> + Send + Sync>,
}

impl DocstoreFn {
    pub fn new<F>(lookup: F) -> Self
    where
        F: Fn(&str) -> Option<Document> + Send + Sync + 'static,
    {
        Self {
            lookup: Arc::new(lookup),
        }
    }

    pub fn search(&self, key: &str) -> DocstoreSearchResult {
        (self.lookup)(key).into()
    }
}

impl fmt::Debug for DocstoreFn {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DocstoreFn(..)")
    }
}

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
