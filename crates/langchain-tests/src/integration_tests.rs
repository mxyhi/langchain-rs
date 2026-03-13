pub mod base_store;
pub mod cache;
pub mod indexer;
pub mod tools;

use langchain_core::documents::Document;
use langchain_core::messages::BaseMessage;

use crate::{
    ChatModelUnitHarness, ChatModelUnitTests, RetrieverUnderTest, VectorStoreUnderTest,
    assert_chat_model_batch, assert_get_by_ids_behavior,
    assert_retriever_returns_expected_document, assert_similarity_search_finds_expected_document,
};

pub trait ChatModelIntegrationHarness: ChatModelUnitHarness {
    fn prompts(&self) -> Vec<Vec<BaseMessage>>;

    fn expected_batch_responses(&self) -> Vec<&'static str>;
}

pub struct ChatModelIntegrationTests<H> {
    harness: H,
}

impl<H> ChatModelIntegrationTests<H> {
    pub fn new(harness: H) -> Self {
        Self { harness }
    }
}

impl<H> ChatModelIntegrationTests<H>
where
    H: ChatModelIntegrationHarness,
{
    pub async fn run(&self) {
        ChatModelUnitTests::new(&self.harness).run().await;

        let model = self.harness.model();
        let expected = self.harness.expected_batch_responses();
        assert_chat_model_batch(&model, self.harness.prompts(), &expected).await;
    }
}

impl<H> ChatModelUnitHarness for &H
where
    H: ChatModelUnitHarness,
{
    type Model = H::Model;

    fn model(&self) -> Self::Model {
        (*self).model()
    }

    fn prompt(&self) -> Vec<BaseMessage> {
        (*self).prompt()
    }

    fn expected_response(&self) -> &'static str {
        (*self).expected_response()
    }

    fn expected_usage_tokens(&self) -> Option<(usize, usize)> {
        (*self).expected_usage_tokens()
    }
}

pub trait RetrieverIntegrationHarness {
    type Retriever: RetrieverUnderTest;

    fn retriever(&self) -> Self::Retriever;

    fn query(&self) -> &'static str;

    fn expected_page_content(&self) -> &'static str;
}

pub struct RetrieverIntegrationTests<H> {
    harness: H,
}

impl<H> RetrieverIntegrationTests<H> {
    pub fn new(harness: H) -> Self {
        Self { harness }
    }
}

impl<H> RetrieverIntegrationTests<H>
where
    H: RetrieverIntegrationHarness,
{
    pub async fn run(&self) {
        let retriever = self.harness.retriever();
        assert_retriever_returns_expected_document(
            &retriever,
            self.harness.query(),
            self.harness.expected_page_content(),
        )
        .await;
    }
}

pub trait VectorStoreIntegrationHarness {
    type Store: VectorStoreUnderTest;

    fn store(&self) -> Self::Store;

    fn documents(&self) -> Vec<Document>;

    fn query(&self) -> &'static str;

    fn expected_page_content(&self) -> &'static str;

    fn requested_ids(&self) -> Vec<String>;

    fn get_by_ids_should_be_supported(&self) -> bool {
        true
    }
}

pub struct VectorStoreIntegrationTests<H> {
    harness: H,
}

impl<H> VectorStoreIntegrationTests<H> {
    pub fn new(harness: H) -> Self {
        Self { harness }
    }
}

impl<H> VectorStoreIntegrationTests<H>
where
    H: VectorStoreIntegrationHarness,
{
    pub async fn run(&self) {
        let documents = self.harness.documents();

        let mut store = self.harness.store();
        assert_similarity_search_finds_expected_document(
            &mut store,
            self.harness.query(),
            documents.clone(),
            self.harness.expected_page_content(),
        )
        .await;

        let mut store = self.harness.store();
        assert_get_by_ids_behavior(
            &mut store,
            documents,
            self.harness.requested_ids(),
            self.harness.get_by_ids_should_be_supported(),
        )
        .await;
    }
}

pub use base_store::{
    BaseStoreAsyncHarness, BaseStoreAsyncTests, BaseStoreHarness, BaseStoreSyncTests,
};
pub use cache::{AsyncCacheHarness, AsyncCacheTestSuite, SyncCacheHarness, SyncCacheTestSuite};
pub use indexer::{DocumentIndexHarness, DocumentIndexIntegrationTests};
pub use tools::{ToolIntegrationHarness, ToolsIntegrationTests};
