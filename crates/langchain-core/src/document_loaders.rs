use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use futures_util::future::BoxFuture;
use serde_json::Value;
use tokio::fs;

use crate::LangChainError;
use crate::documents::Document;

pub type PathLike = PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub struct Blob {
    data: Vec<u8>,
    path: Option<PathBuf>,
    mime_type: Option<String>,
    metadata: BTreeMap<String, Value>,
}

impl Blob {
    pub fn from_bytes(data: impl Into<Vec<u8>>) -> Self {
        Self {
            data: data.into(),
            path: None,
            mime_type: None,
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn with_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }

    pub fn with_metadata(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn mime_type(&self) -> Option<&str> {
        self.mime_type.as_deref()
    }

    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }

    pub fn as_utf8(&self) -> Result<String, LangChainError> {
        String::from_utf8(self.data.clone())
            .map_err(|error| LangChainError::request(format!("blob is not valid utf-8: {error}")))
    }
}

pub trait BaseBlobParser: Send + Sync {
    fn parse<'a>(&'a self, blob: &'a Blob) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>>;
}

pub trait BlobLoader: Send + Sync {
    fn load_blobs<'a>(&'a self) -> BoxFuture<'a, Result<Vec<Blob>, LangChainError>>;
}

#[derive(Debug, Clone, Default)]
pub struct TextBlobParser;

impl BaseBlobParser for TextBlobParser {
    fn parse<'a>(&'a self, blob: &'a Blob) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        Box::pin(async move {
            let mut metadata = blob.metadata().clone();
            if let Some(path) = blob.path() {
                metadata.insert(
                    "source".to_owned(),
                    Value::String(path.display().to_string()),
                );
            }
            if let Some(mime_type) = blob.mime_type() {
                metadata.insert("mime_type".to_owned(), Value::String(mime_type.to_owned()));
            }

            Ok(vec![Document {
                page_content: blob.as_utf8()?,
                metadata,
                id: None,
            }])
        })
    }
}

pub trait BaseLoader: Send + Sync {
    fn load<'a>(&'a self) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>>;
}

#[derive(Debug, Clone)]
pub struct TextLoader {
    path: PathBuf,
    metadata: BTreeMap<String, Value>,
    parser: TextBlobParser,
}

impl TextLoader {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            metadata: BTreeMap::new(),
            parser: TextBlobParser,
        }
    }

    pub fn with_metadata(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl BaseLoader for TextLoader {
    fn load<'a>(&'a self) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        Box::pin(async move {
            let data = fs::read(&self.path).await.map_err(|error| {
                LangChainError::request(format!("failed to read {}: {error}", self.path.display()))
            })?;

            let blob = Blob::from_bytes(data)
                .with_path(self.path.clone())
                .with_mime_type("text/plain")
                .with_metadata(self.metadata.clone());
            self.parser.parse(&blob).await
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct StaticDocumentLoader {
    documents: Vec<Document>,
}

impl StaticDocumentLoader {
    pub fn new(documents: Vec<Document>) -> Self {
        Self { documents }
    }
}

impl BaseLoader for StaticDocumentLoader {
    fn load<'a>(&'a self) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        Box::pin(async move { Ok(self.documents.clone()) })
    }
}

#[derive(Debug, Clone, Default)]
pub struct StaticBlobLoader {
    blobs: Vec<Blob>,
}

impl StaticBlobLoader {
    pub fn new(blobs: Vec<Blob>) -> Self {
        Self { blobs }
    }
}

impl BlobLoader for StaticBlobLoader {
    fn load_blobs<'a>(&'a self) -> BoxFuture<'a, Result<Vec<Blob>, LangChainError>> {
        Box::pin(async move { Ok(self.blobs.clone()) })
    }
}

#[derive(Debug, Clone, Default)]
pub struct LangSmithLoader {
    documents: Vec<Document>,
}

impl LangSmithLoader {
    pub fn new(documents: Vec<Document>) -> Self {
        Self { documents }
    }
}

impl BaseLoader for LangSmithLoader {
    fn load<'a>(&'a self) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        Box::pin(async move { Ok(self.documents.clone()) })
    }
}
