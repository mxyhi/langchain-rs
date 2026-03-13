use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::caches::InMemoryCache;
use langchain_core::documents::Document;
use langchain_core::indexing::{DeleteResponse, DocumentIndex, UpsertResponse};
use langchain_core::messages::ToolCall;
use langchain_core::retrievers::BaseRetriever;
use langchain_core::runnables::RunnableConfig;
use langchain_core::stores::InMemoryStore;
use langchain_core::tools::{Tool, ToolDefinition};
use langchain_tests::integration_tests::{
    AsyncCacheHarness, AsyncCacheTestSuite, BaseStoreAsyncHarness, BaseStoreAsyncTests,
    BaseStoreHarness, BaseStoreSyncTests, DocumentIndexHarness, DocumentIndexIntegrationTests,
    SyncCacheHarness, SyncCacheTestSuite, ToolIntegrationHarness, ToolsIntegrationTests,
};
use langchain_tests::unit_tests::{ToolUnitHarness, ToolsUnitTests};

#[derive(Clone, Copy, Default)]
struct CacheHarness;

impl SyncCacheHarness for CacheHarness {
    type Cache = InMemoryCache;

    fn cache(&self) -> Self::Cache {
        InMemoryCache::new()
    }
}

impl AsyncCacheHarness for CacheHarness {}

#[derive(Clone, Copy, Default)]
struct StoreHarness;

impl BaseStoreHarness for StoreHarness {
    type Store = InMemoryStore<String>;
    type Value = String;

    fn store(&self) -> Self::Store {
        InMemoryStore::new()
    }

    fn three_values(&self) -> (Self::Value, Self::Value, Self::Value) {
        ("foo".to_owned(), "bar".to_owned(), "baz".to_owned())
    }
}

impl BaseStoreAsyncHarness for StoreHarness {}

#[derive(Clone, Copy, Default)]
struct ToolHarness;

impl ToolUnitHarness for ToolHarness {
    type Tool = Tool;

    fn tool(&self) -> Self::Tool {
        Tool::new(ToolDefinition::new("lookup", "Look up a record"), |input| {
            Box::pin(async move { Ok(format!("result:{input}")) })
        })
    }

    fn example_tool_call(&self) -> ToolCall {
        ToolCall::new("lookup", serde_json::json!({ "input": "rust" })).with_id("call_lookup_1")
    }
}

impl ToolIntegrationHarness for ToolHarness {}

#[derive(Default)]
struct FakeDocumentIndex {
    documents: BTreeMap<String, Document>,
}

impl BaseRetriever for FakeDocumentIndex {
    fn get_relevant_documents<'a>(
        &'a self,
        query: &'a str,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        Box::pin(async move {
            Ok(self
                .documents
                .values()
                .filter(|document| document.page_content.contains(query))
                .cloned()
                .collect())
        })
    }
}

impl DocumentIndex for FakeDocumentIndex {
    fn upsert_documents<'a>(
        &'a mut self,
        documents: Vec<Document>,
    ) -> BoxFuture<'a, Result<UpsertResponse, LangChainError>> {
        Box::pin(async move {
            let added = documents.len();
            for (index, document) in documents.into_iter().enumerate() {
                let id = document
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("doc-{}", self.documents.len() + index + 1));
                let mut stored = document;
                stored.id = Some(id.clone());
                self.documents.insert(id, stored);
            }

            Ok(UpsertResponse {
                num_added: added,
                num_skipped: 0,
            })
        })
    }

    fn delete_documents<'a>(
        &'a mut self,
        ids: Vec<String>,
    ) -> BoxFuture<'a, Result<DeleteResponse, LangChainError>> {
        Box::pin(async move {
            let mut deleted = 0;
            for id in ids {
                if self.documents.remove(&id).is_some() {
                    deleted += 1;
                }
            }
            Ok(DeleteResponse {
                num_deleted: deleted,
            })
        })
    }
}

#[derive(Clone, Copy, Default)]
struct DocumentIndexHarnessImpl;

impl DocumentIndexHarness for DocumentIndexHarnessImpl {
    type Index = FakeDocumentIndex;

    fn index(&self) -> Self::Index {
        FakeDocumentIndex::default()
    }

    fn seed_documents(&self) -> Vec<Document> {
        vec![
            Document {
                page_content: "alpha rust".to_owned(),
                metadata: BTreeMap::new(),
                id: Some("doc-1".to_owned()),
            },
            Document {
                page_content: "beta langchain".to_owned(),
                metadata: BTreeMap::new(),
                id: Some("doc-2".to_owned()),
            },
        ]
    }

    fn overwrite_document(&self) -> Document {
        Document {
            page_content: "alpha rust updated".to_owned(),
            metadata: BTreeMap::new(),
            id: Some("doc-1".to_owned()),
        }
    }

    fn query(&self) -> &'static str {
        "rust"
    }

    fn expected_page_content(&self) -> &'static str {
        "alpha rust"
    }
}

#[test]
fn cache_harnesses_cover_sync_and_async_cache_contracts() {
    SyncCacheTestSuite::new(CacheHarness).run();
}

#[tokio::test]
async fn async_cache_harness_covers_async_cache_contract() {
    AsyncCacheTestSuite::new(CacheHarness).run().await;
}

#[test]
fn base_store_sync_harness_covers_store_contract() {
    BaseStoreSyncTests::new(StoreHarness).run();
}

#[tokio::test]
async fn base_store_async_harness_covers_store_contract() {
    BaseStoreAsyncTests::new(StoreHarness).run().await;
}

#[tokio::test]
async fn tool_harnesses_cover_unit_and_integration_contracts() {
    ToolsUnitTests::new(ToolHarness).run().await;
    ToolsIntegrationTests::new(ToolHarness).run().await;
}

#[tokio::test]
async fn document_index_harness_covers_upsert_delete_and_retrieve_contracts() {
    DocumentIndexIntegrationTests::new(DocumentIndexHarnessImpl)
        .run()
        .await;
}
