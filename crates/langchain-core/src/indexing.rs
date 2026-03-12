use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

use futures_util::future::BoxFuture;

use crate::LangChainError;
use crate::documents::Document;
use crate::retrievers::BaseRetriever;
use crate::vectorstores::VectorStore;

pub trait RecordManager: Send + Sync {
    fn create_schema(&mut self) -> Result<(), LangChainError>;

    fn update(
        &mut self,
        keys: Vec<String>,
        group_ids: Option<Vec<String>>,
    ) -> Result<(), LangChainError>;

    fn exists(&self, keys: Vec<String>) -> Result<Vec<bool>, LangChainError>;

    fn list_keys(
        &self,
        before: Option<f64>,
        after: Option<f64>,
        group_ids: Option<Vec<String>>,
        limit: Option<usize>,
    ) -> Result<Vec<String>, LangChainError>;

    fn delete_keys(&mut self, keys: Vec<String>) -> Result<(), LangChainError>;

    fn acreate_schema<'a>(&'a mut self) -> BoxFuture<'a, Result<(), LangChainError>> {
        Box::pin(async move { self.create_schema() })
    }

    fn aupdate<'a>(
        &'a mut self,
        keys: Vec<String>,
        group_ids: Option<Vec<String>>,
    ) -> BoxFuture<'a, Result<(), LangChainError>> {
        Box::pin(async move { self.update(keys, group_ids) })
    }

    fn aexists<'a>(
        &'a self,
        keys: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<bool>, LangChainError>> {
        Box::pin(async move { self.exists(keys) })
    }

    fn alist_keys<'a>(
        &'a self,
        before: Option<f64>,
        after: Option<f64>,
        group_ids: Option<Vec<String>>,
        limit: Option<usize>,
    ) -> BoxFuture<'a, Result<Vec<String>, LangChainError>> {
        Box::pin(async move { self.list_keys(before, after, group_ids, limit) })
    }

    fn adelete_keys<'a>(
        &'a mut self,
        keys: Vec<String>,
    ) -> BoxFuture<'a, Result<(), LangChainError>> {
        Box::pin(async move { self.delete_keys(keys) })
    }
}

#[derive(Debug, Clone, PartialEq)]
struct RecordEntry {
    updated_at: f64,
    group_ids: BTreeSet<String>,
}

#[derive(Debug, Clone)]
pub struct InMemoryRecordManager {
    namespace: String,
    schema_created: bool,
    records: BTreeMap<String, RecordEntry>,
}

impl InMemoryRecordManager {
    pub fn new(namespace: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            schema_created: false,
            records: BTreeMap::new(),
        }
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    fn get_time(&self) -> Result<f64, LangChainError> {
        Ok(SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| LangChainError::request(format!("system clock error: {error}")))?
            .as_secs_f64())
    }
}

impl RecordManager for InMemoryRecordManager {
    fn create_schema(&mut self) -> Result<(), LangChainError> {
        self.schema_created = true;
        Ok(())
    }

    fn update(
        &mut self,
        keys: Vec<String>,
        group_ids: Option<Vec<String>>,
    ) -> Result<(), LangChainError> {
        if !self.schema_created {
            self.create_schema()?;
        }

        let timestamp = self.get_time()?;
        let groups = group_ids
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>();

        for key in keys {
            self.records.insert(
                key,
                RecordEntry {
                    updated_at: timestamp,
                    group_ids: groups.clone(),
                },
            );
        }
        Ok(())
    }

    fn exists(&self, keys: Vec<String>) -> Result<Vec<bool>, LangChainError> {
        Ok(keys
            .into_iter()
            .map(|key| self.records.contains_key(&key))
            .collect())
    }

    fn list_keys(
        &self,
        before: Option<f64>,
        after: Option<f64>,
        group_ids: Option<Vec<String>>,
        limit: Option<usize>,
    ) -> Result<Vec<String>, LangChainError> {
        let requested_groups = group_ids
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>();

        let mut keys = self
            .records
            .iter()
            .filter(|(_, entry)| before.is_none_or(|value| entry.updated_at < value))
            .filter(|(_, entry)| after.is_none_or(|value| entry.updated_at > value))
            .filter(|(_, entry)| {
                requested_groups.is_empty()
                    || entry
                        .group_ids
                        .iter()
                        .any(|group| requested_groups.contains(group))
            })
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();

        if let Some(limit) = limit {
            keys.truncate(limit);
        }
        Ok(keys)
    }

    fn delete_keys(&mut self, keys: Vec<String>) -> Result<(), LangChainError> {
        for key in keys {
            self.records.remove(&key);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UpsertResponse {
    pub num_added: usize,
    pub num_skipped: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeleteResponse {
    pub num_deleted: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IndexingResult {
    pub num_added: usize,
    pub num_skipped: usize,
    pub num_deleted: usize,
}

pub trait DocumentIndex: BaseRetriever + Send + Sync {
    fn upsert_documents<'a>(
        &'a mut self,
        documents: Vec<Document>,
    ) -> BoxFuture<'a, Result<UpsertResponse, LangChainError>>;

    fn delete_documents<'a>(
        &'a mut self,
        ids: Vec<String>,
    ) -> BoxFuture<'a, Result<DeleteResponse, LangChainError>>;
}

pub async fn index<V, R>(
    documents: Vec<Document>,
    vector_store: &mut V,
    record_manager: &mut R,
    cleanup: bool,
) -> Result<IndexingResult, LangChainError>
where
    V: VectorStore + Send + Sync,
    R: RecordManager,
{
    record_manager.acreate_schema().await?;

    let mut live_keys = BTreeSet::new();
    let mut prepared = Vec::new();
    for mut document in documents {
        let key = document
            .id
            .clone()
            .unwrap_or_else(|| document_key(&document));
        document.id = Some(key.clone());
        live_keys.insert(key);
        prepared.push(document);
    }

    let keys = prepared
        .iter()
        .map(|document| document.id.clone().expect("prepared documents have ids"))
        .collect::<Vec<_>>();
    let exists = record_manager.aexists(keys.clone()).await?;

    let mut to_add = Vec::new();
    let mut new_keys = Vec::new();
    let mut skipped = 0;
    for (document, exists) in prepared.into_iter().zip(exists) {
        let key = document
            .id
            .clone()
            .expect("prepared documents keep their ids");
        if exists {
            skipped += 1;
        } else {
            new_keys.push(key);
            to_add.push(document);
        }
    }

    let added = if to_add.is_empty() {
        0
    } else {
        vector_store.add_documents(to_add).await?.len()
    };
    if !new_keys.is_empty() {
        record_manager.aupdate(new_keys, None).await?;
    }

    let mut deleted = 0;
    if cleanup {
        let stale_keys = record_manager
            .alist_keys(None, None, None, None)
            .await?
            .into_iter()
            .filter(|key| !live_keys.contains(key))
            .collect::<Vec<_>>();
        if !stale_keys.is_empty() {
            vector_store.delete(&stale_keys)?;
            deleted = stale_keys.len();
            record_manager.adelete_keys(stale_keys).await?;
        }
    }

    Ok(IndexingResult {
        num_added: added,
        num_skipped: skipped,
        num_deleted: deleted,
    })
}

pub async fn aindex<V, R>(
    documents: Vec<Document>,
    vector_store: &mut V,
    record_manager: &mut R,
    cleanup: bool,
) -> Result<IndexingResult, LangChainError>
where
    V: VectorStore + Send + Sync,
    R: RecordManager,
{
    index(documents, vector_store, record_manager, cleanup).await
}

fn document_key(document: &Document) -> String {
    let mut hasher = DefaultHasher::new();
    document.page_content.hash(&mut hasher);
    let metadata = serde_json::to_string(&document.metadata).unwrap_or_else(|_| String::from("{}"));
    metadata.hash(&mut hasher);
    format!("doc_{:x}", hasher.finish())
}
