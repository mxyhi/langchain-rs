use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::documents::Document;
use langchain_core::language_models::{ParrotChatModel, ParrotLLM};
use langchain_core::messages::{BaseMessage, HumanMessage};
use langchain_tests::integration_tests::{
    ChatModelIntegrationHarness, ChatModelIntegrationTests, RetrieverIntegrationHarness,
    RetrieverIntegrationTests, VectorStoreIntegrationHarness, VectorStoreIntegrationTests,
};
use langchain_tests::unit_tests::{
    ChatModelUnitHarness, ChatModelUnitTests, EmbeddingsUnitHarness, EmbeddingsUnitTests,
    LlmUnitHarness, LlmUnitTests,
};
use langchain_tests::{
    EmbeddingsUnderTest, RetrieverUnderTest, SimilaritySearchResult, VectorStoreUnderTest,
};

#[derive(Clone, Copy)]
struct ParrotChatHarness;

impl ChatModelUnitHarness for ParrotChatHarness {
    type Model = ParrotChatModel;

    fn model(&self) -> Self::Model {
        ParrotChatModel::new("parrot-1", 5)
    }

    fn prompt(&self) -> Vec<BaseMessage> {
        vec![BaseMessage::from(HumanMessage::new("hello"))]
    }

    fn expected_response(&self) -> &'static str {
        "hello"
    }

    fn expected_usage_tokens(&self) -> Option<(usize, usize)> {
        Some((5, 5))
    }
}

impl ChatModelIntegrationHarness for ParrotChatHarness {
    fn prompts(&self) -> Vec<Vec<BaseMessage>> {
        vec![self.prompt(), self.prompt()]
    }

    fn expected_batch_responses(&self) -> Vec<&'static str> {
        vec!["hello", "hello"]
    }
}

struct FakeEmbeddings;

impl EmbeddingsUnderTest for FakeEmbeddings {
    fn embed_query<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>, LangChainError>> {
        Box::pin(async move {
            Ok(vec![
                text.len() as f32,
                text.bytes().map(f32::from).sum::<f32>(),
                text.chars().count() as f32,
            ])
        })
    }

    fn embed_documents<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, LangChainError>> {
        Box::pin(async move {
            Ok(texts
                .into_iter()
                .map(|text| {
                    vec![
                        text.len() as f32,
                        text.bytes().map(f32::from).sum::<f32>(),
                        text.chars().count() as f32,
                    ]
                })
                .collect())
        })
    }
}

#[derive(Clone, Copy)]
struct EmbeddingsHarness;

impl EmbeddingsUnitHarness for EmbeddingsHarness {
    type Embeddings = FakeEmbeddings;

    fn embeddings(&self) -> Self::Embeddings {
        FakeEmbeddings
    }

    fn query(&self) -> &'static str {
        "alpha"
    }

    fn documents(&self) -> Vec<String> {
        vec!["beta".to_owned(), "gamma".to_owned()]
    }

    fn expected_dimension(&self) -> usize {
        3
    }
}

#[derive(Clone, Copy)]
struct ParrotLlmHarness;

impl LlmUnitHarness for ParrotLlmHarness {
    type Model = ParrotLLM;

    fn model(&self) -> Self::Model {
        ParrotLLM::new("parrot-llm", 5)
    }

    fn prompt(&self) -> &'static str {
        "alphabet"
    }

    fn expected_response(&self) -> &'static str {
        "alpha"
    }

    fn prompts(&self) -> Vec<String> {
        vec!["alphabet".to_owned(), "beta".to_owned()]
    }

    fn expected_texts(&self) -> Vec<&'static str> {
        vec!["alpha", "beta"]
    }

    fn expected_token_usage(&self) -> Option<(usize, usize)> {
        Some((12, 9))
    }
}

#[derive(Default)]
struct FakeRetriever;

impl RetrieverUnderTest for FakeRetriever {
    fn invoke<'a>(
        &'a self,
        query: &'a str,
    ) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        Box::pin(async move {
            Ok(vec![Document {
                page_content: format!("hit:{query}"),
                metadata: BTreeMap::new(),
                id: Some("doc-1".to_owned()),
            }])
        })
    }
}

#[derive(Clone, Copy)]
struct RetrieverHarness;

impl RetrieverIntegrationHarness for RetrieverHarness {
    type Retriever = FakeRetriever;

    fn retriever(&self) -> Self::Retriever {
        FakeRetriever
    }

    fn query(&self) -> &'static str {
        "rust"
    }

    fn expected_page_content(&self) -> &'static str {
        "hit:rust"
    }
}

#[derive(Default)]
struct FakeVectorStore {
    documents: BTreeMap<String, Document>,
}

impl VectorStoreUnderTest for FakeVectorStore {
    fn add_documents<'a>(
        &'a mut self,
        documents: Vec<Document>,
    ) -> BoxFuture<'a, Result<Vec<String>, LangChainError>> {
        Box::pin(async move {
            let mut ids = Vec::with_capacity(documents.len());
            for document in documents {
                let id = document
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("doc-{}", self.documents.len() + 1));
                let mut stored = document;
                stored.id = Some(id.clone());
                self.documents.insert(id.clone(), stored);
                ids.push(id);
            }
            Ok(ids)
        })
    }

    fn similarity_search<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> BoxFuture<'a, Result<Vec<SimilaritySearchResult>, LangChainError>> {
        Box::pin(async move {
            Ok(self
                .documents
                .values()
                .filter(|document| document.page_content.contains(query))
                .take(limit)
                .cloned()
                .map(|document| SimilaritySearchResult {
                    document,
                    score: Some(1.0),
                })
                .collect())
        })
    }

    fn get_by_ids<'a>(
        &'a self,
        ids: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        Box::pin(async move {
            Ok(ids
                .into_iter()
                .filter_map(|id| self.documents.get(&id).cloned())
                .collect())
        })
    }
}

#[derive(Clone, Copy)]
struct VectorStoreHarness;

impl VectorStoreIntegrationHarness for VectorStoreHarness {
    type Store = FakeVectorStore;

    fn store(&self) -> Self::Store {
        FakeVectorStore::default()
    }

    fn documents(&self) -> Vec<Document> {
        vec![
            Document {
                page_content: "hello world".to_owned(),
                metadata: BTreeMap::new(),
                id: Some("doc-1".to_owned()),
            },
            Document {
                page_content: "goodbye moon".to_owned(),
                metadata: BTreeMap::new(),
                id: Some("doc-2".to_owned()),
            },
        ]
    }

    fn query(&self) -> &'static str {
        "hello"
    }

    fn expected_page_content(&self) -> &'static str {
        "hello world"
    }

    fn requested_ids(&self) -> Vec<String> {
        vec!["doc-1".to_owned(), "doc-2".to_owned()]
    }
}

#[tokio::test]
async fn standard_harnesses_run_chat_model_suites() {
    ChatModelUnitTests::new(ParrotChatHarness).run().await;
    ChatModelIntegrationTests::new(ParrotChatHarness)
        .run()
        .await;
}

#[tokio::test]
async fn standard_harnesses_run_embeddings_and_llm_suites() {
    EmbeddingsUnitTests::new(EmbeddingsHarness).run().await;
    LlmUnitTests::new(ParrotLlmHarness).run().await;
}

#[tokio::test]
async fn standard_harnesses_run_retriever_and_vectorstore_suites() {
    RetrieverIntegrationTests::new(RetrieverHarness).run().await;
    VectorStoreIntegrationTests::new(VectorStoreHarness)
        .run()
        .await;
}

#[tokio::test]
async fn namespaced_modules_match_reference_layout() {
    langchain_tests::unit_tests::chat_models::ChatModelUnitTests::new(ParrotChatHarness)
        .run()
        .await;
    langchain_tests::unit_tests::embeddings::EmbeddingsUnitTests::new(EmbeddingsHarness)
        .run()
        .await;
    langchain_tests::integration_tests::chat_models::ChatModelIntegrationTests::new(
        ParrotChatHarness,
    )
    .run()
    .await;
    langchain_tests::integration_tests::embeddings::EmbeddingsIntegrationTests::new(
        EmbeddingsHarness,
    )
    .run()
    .await;
    langchain_tests::integration_tests::retrievers::RetrieversIntegrationTests::new(
        RetrieverHarness,
    )
    .run()
    .await;
    langchain_tests::integration_tests::vectorstores::VectorStoreIntegrationTests::new(
        VectorStoreHarness,
    )
    .run()
    .await;
}
