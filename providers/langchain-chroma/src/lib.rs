use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::documents::Document;
use langchain_core::embeddings::Embeddings;
use langchain_core::vectorstores::{InMemoryVectorStore, VectorStore};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::runtime::{Handle, Runtime};

const DEFAULT_TENANT: &str = "default_tenant";
const DEFAULT_DATABASE: &str = "default_database";

pub struct Chroma<E> {
    collection_name: String,
    backend: Backend<E>,
    next_id: usize,
}

enum Backend<E> {
    Local(InMemoryVectorStore<E>),
    Remote(RemoteChroma<E>),
}

struct RemoteChroma<E> {
    embedding: Arc<E>,
    client: Client,
    base_url: String,
    tenant: String,
    database: String,
    collection_id: Mutex<Option<String>>,
}

#[derive(Debug, Deserialize)]
struct CollectionResponse {
    id: String,
}

#[derive(Debug, Deserialize)]
struct QueryResponse {
    documents: Vec<Vec<String>>,
    metadatas: Vec<Vec<BTreeMap<String, Value>>>,
    ids: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct GetResponse {
    documents: Vec<String>,
    metadatas: Vec<BTreeMap<String, Value>>,
    ids: Vec<String>,
}

impl<E> Chroma<E>
where
    E: Embeddings,
{
    pub fn new(collection_name: impl Into<String>, embedding: E) -> Self {
        Self {
            collection_name: collection_name.into(),
            backend: Backend::Local(InMemoryVectorStore::new(embedding)),
            next_id: 0,
        }
    }

    pub fn new_remote(
        collection_name: impl Into<String>,
        base_url: impl Into<String>,
        embedding: E,
    ) -> Self {
        Self::new_remote_with_namespace(
            collection_name,
            base_url,
            DEFAULT_TENANT,
            DEFAULT_DATABASE,
            embedding,
        )
    }

    pub fn new_remote_with_namespace(
        collection_name: impl Into<String>,
        base_url: impl Into<String>,
        tenant: impl Into<String>,
        database: impl Into<String>,
        embedding: E,
    ) -> Self {
        Self {
            collection_name: collection_name.into(),
            backend: Backend::Remote(RemoteChroma {
                embedding: Arc::new(embedding),
                client: Client::new(),
                base_url: base_url.into().trim_end_matches('/').to_owned(),
                tenant: tenant.into(),
                database: database.into(),
                collection_id: Mutex::new(None),
            }),
            next_id: 0,
        }
    }

    pub fn collection_name(&self) -> &str {
        &self.collection_name
    }

    pub fn embedding(&self) -> &E {
        match &self.backend {
            Backend::Local(inner) => inner.embedding(),
            Backend::Remote(remote) => remote.embedding.as_ref(),
        }
    }

    async fn similarity_search_remote(
        &self,
        remote: &RemoteChroma<E>,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Document>, LangChainError> {
        let collection_id = Self::ensure_collection_id(&self.collection_name, remote).await?;
        let query_embedding = remote.embedding.embed_query(query).await?;
        let response = remote
            .client
            .post(format!(
                "{}/api/v2/tenants/{}/databases/{}/collections/{}/query",
                remote.base_url, remote.tenant, remote.database, collection_id
            ))
            .json(&json!({
                "query_embeddings": [query_embedding],
                "n_results": limit,
                "include": ["documents", "metadatas", "distances"],
            }))
            .send()
            .await
            .map_err(http_error("query chroma records"))?
            .error_for_status()
            .map_err(http_error("query chroma records"))?
            .json::<QueryResponse>()
            .await
            .map_err(http_error("decode chroma query response"))?;

        Ok(response
            .documents
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter()
            .zip(
                response
                    .metadatas
                    .into_iter()
                    .next()
                    .unwrap_or_default()
                    .into_iter(),
            )
            .zip(
                response
                    .ids
                    .into_iter()
                    .next()
                    .unwrap_or_default()
                    .into_iter(),
            )
            .map(|((page_content, metadata), id)| Document {
                page_content,
                metadata,
                id: Some(id),
            })
            .collect())
    }

    async fn get_by_ids_remote(
        &self,
        remote: &RemoteChroma<E>,
        ids: &[String],
    ) -> Result<Vec<Document>, LangChainError> {
        let collection_id = Self::ensure_collection_id(&self.collection_name, remote).await?;
        let response = remote
            .client
            .post(format!(
                "{}/api/v2/tenants/{}/databases/{}/collections/{}/get",
                remote.base_url, remote.tenant, remote.database, collection_id
            ))
            .json(&json!({
                "ids": ids,
                "include": ["documents", "metadatas"],
            }))
            .send()
            .await
            .map_err(http_error("get chroma records"))?
            .error_for_status()
            .map_err(http_error("get chroma records"))?
            .json::<GetResponse>()
            .await
            .map_err(http_error("decode chroma get response"))?;

        Ok(response
            .documents
            .into_iter()
            .zip(response.metadatas.into_iter())
            .zip(response.ids.into_iter())
            .map(|((page_content, metadata), id)| Document {
                page_content,
                metadata,
                id: Some(id),
            })
            .collect())
    }

    async fn delete_remote(
        collection_name: &str,
        remote: &RemoteChroma<E>,
        ids: &[String],
    ) -> Result<bool, LangChainError> {
        let collection_id = Self::ensure_collection_id(collection_name, remote).await?;
        remote
            .client
            .post(format!(
                "{}/api/v2/tenants/{}/databases/{}/collections/{}/delete",
                remote.base_url, remote.tenant, remote.database, collection_id
            ))
            .json(&json!({ "ids": ids }))
            .send()
            .await
            .map_err(http_error("delete chroma records"))?
            .error_for_status()
            .map_err(http_error("delete chroma records"))?;
        Ok(true)
    }

    async fn ensure_collection_id(
        collection_name: &str,
        remote: &RemoteChroma<E>,
    ) -> Result<String, LangChainError> {
        if let Some(collection_id) = remote
            .collection_id
            .lock()
            .expect("collection mutex")
            .clone()
        {
            return Ok(collection_id);
        }

        let response = remote
            .client
            .post(format!(
                "{}/api/v2/tenants/{}/databases/{}/collections",
                remote.base_url, remote.tenant, remote.database
            ))
            .json(&json!({
                "name": collection_name,
                "get_or_create": true,
            }))
            .send()
            .await
            .map_err(http_error("create chroma collection"))?
            .error_for_status()
            .map_err(http_error("create chroma collection"))?
            .json::<CollectionResponse>()
            .await
            .map_err(http_error("decode chroma collection response"))?;

        let mut guard = remote.collection_id.lock().expect("collection mutex");
        if let Some(existing) = guard.clone() {
            return Ok(existing);
        }
        *guard = Some(response.id.clone());
        Ok(response.id)
    }
}

impl<E> VectorStore for Chroma<E>
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
                    let collection_id =
                        Self::ensure_collection_id(&self.collection_name, remote).await?;
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
                    let metadatas = documents
                        .iter()
                        .map(|document| document.metadata.clone())
                        .collect::<Vec<_>>();
                    let embeddings = remote.embedding.embed_documents(texts.clone()).await?;

                    remote
                        .client
                        .post(format!(
                            "{}/api/v2/tenants/{}/databases/{}/collections/{}/add",
                            remote.base_url, remote.tenant, remote.database, collection_id
                        ))
                        .json(&json!({
                            "ids": ids,
                            "documents": texts,
                            "metadatas": metadatas,
                            "embeddings": embeddings,
                        }))
                        .send()
                        .await
                        .map_err(http_error("add chroma records"))?
                        .error_for_status()
                        .map_err(http_error("add chroma records"))?;

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
        let collection_name = self.collection_name.clone();
        match &mut self.backend {
            Backend::Local(inner) => inner.delete(ids),
            Backend::Remote(remote) => {
                block_on_sync(Self::delete_remote(&collection_name, remote, ids))
            }
        }
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

pub mod vectorstores {
    pub use crate::Chroma;
}
