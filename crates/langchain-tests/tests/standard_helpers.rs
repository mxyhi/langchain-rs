use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::documents::Document;
use langchain_core::language_models::{BaseChatModel, ParrotChatModel, ParrotLLM};
use langchain_core::messages::{BaseMessage, HumanMessage};
use langchain_core::runnables::RunnableConfig;
use langchain_tests::{
    EmbeddingsUnderTest, SimilaritySearchResult, VectorStoreUnderTest, assert_chat_model_batch,
    assert_chat_model_response, assert_document_embeddings_count, assert_embedding_dimension,
    assert_get_by_ids_behavior, assert_llm_generate_texts, assert_llm_invoke_response,
    assert_llm_token_usage, assert_query_and_documents_share_dimension,
    assert_similarity_search_finds_expected_document, assert_usage_tokens,
};

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
            let mut matches = self
                .documents
                .values()
                .filter(|document| document.page_content.contains(query))
                .take(limit)
                .cloned()
                .map(|document| SimilaritySearchResult {
                    document,
                    score: Some(1.0),
                })
                .collect::<Vec<_>>();

            if matches.is_empty() {
                matches = self
                    .documents
                    .values()
                    .take(limit)
                    .cloned()
                    .map(|document| SimilaritySearchResult {
                        document,
                        score: Some(0.0),
                    })
                    .collect();
            }

            Ok(matches)
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

#[derive(Default)]
struct WithoutGetByIdsVectorStore(FakeVectorStore);

impl VectorStoreUnderTest for WithoutGetByIdsVectorStore {
    fn add_documents<'a>(
        &'a mut self,
        documents: Vec<Document>,
    ) -> BoxFuture<'a, Result<Vec<String>, LangChainError>> {
        self.0.add_documents(documents)
    }

    fn similarity_search<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> BoxFuture<'a, Result<Vec<SimilaritySearchResult>, LangChainError>> {
        self.0.similarity_search(query, limit)
    }
}

#[tokio::test]
async fn chat_model_helpers_validate_response_and_batch() {
    let model = ParrotChatModel::new("parrot-1", 5);
    let prompt = vec![BaseMessage::from(HumanMessage::new("hello"))];

    assert_chat_model_response(&model, prompt.clone(), "hello").await;
    assert_chat_model_batch(&model, vec![prompt.clone(), prompt], &["hello", "hello"]).await;

    let generated = model
        .generate(
            vec![BaseMessage::from(HumanMessage::new("hello"))],
            RunnableConfig::default(),
        )
        .await
        .expect("generate should succeed");
    assert_usage_tokens(generated.usage_metadata(), 5, 5).expect("usage metadata should match");
}

#[tokio::test]
async fn embedding_helpers_validate_shape() {
    let embeddings = FakeEmbeddings;

    assert_embedding_dimension(&embeddings, "alpha", 3).await;
    assert_document_embeddings_count(
        &embeddings,
        vec!["alpha".to_owned(), "beta".to_owned(), "gamma".to_owned()],
    )
    .await;
    assert_query_and_documents_share_dimension(
        &embeddings,
        "alpha",
        vec!["beta".to_owned(), "gamma".to_owned()],
    )
    .await;
}

#[tokio::test]
async fn llm_helpers_validate_text_and_usage() {
    let model = ParrotLLM::new("parrot-llm", 5);

    assert_llm_invoke_response(&model, "alphabet", "alpha").await;
    let result = assert_llm_generate_texts(
        &model,
        vec!["alphabet".to_owned(), "beta".to_owned()],
        &["alpha", "beta"],
    )
    .await;

    assert_llm_token_usage(result.llm_output(), 12, 9).expect("llm usage metadata should match");
}

#[tokio::test]
async fn vector_store_helpers_cover_similarity_and_get_by_ids() {
    let mut store = FakeVectorStore::default();
    let documents = vec![
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
    ];

    assert_similarity_search_finds_expected_document(
        &mut store,
        "hello",
        documents.clone(),
        "hello world",
    )
    .await;

    let mut supported_store = FakeVectorStore::default();
    assert_get_by_ids_behavior(
        &mut supported_store,
        documents.clone(),
        vec!["doc-1".to_owned(), "doc-2".to_owned()],
        true,
    )
    .await;

    let mut unsupported_store = WithoutGetByIdsVectorStore::default();
    assert_get_by_ids_behavior(
        &mut unsupported_store,
        documents,
        vec!["doc-1".to_owned()],
        false,
    )
    .await;
}
