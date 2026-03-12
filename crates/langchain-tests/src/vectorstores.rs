use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::documents::Document;

#[derive(Debug, Clone)]
pub struct SimilaritySearchResult {
    pub document: Document,
    pub score: Option<f32>,
}

pub trait VectorStoreUnderTest: Send + Sync {
    fn add_documents<'a>(
        &'a mut self,
        documents: Vec<Document>,
    ) -> BoxFuture<'a, Result<Vec<String>, LangChainError>>;

    fn similarity_search<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> BoxFuture<'a, Result<Vec<SimilaritySearchResult>, LangChainError>>;

    fn get_by_ids<'a>(
        &'a self,
        ids: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        let _ = ids;
        Box::pin(async { Err(LangChainError::unsupported("get_by_ids is not implemented")) })
    }
}

pub async fn assert_similarity_search_finds_expected_document<V>(
    store: &mut V,
    query: &str,
    documents: Vec<Document>,
    expected_page_content: &str,
) where
    V: VectorStoreUnderTest,
{
    store
        .add_documents(documents)
        .await
        .expect("vector store add_documents should succeed");

    let results = store
        .similarity_search(query, 4)
        .await
        .expect("similarity_search should succeed");

    assert!(
        results
            .iter()
            .any(|result| result.document.page_content == expected_page_content),
        "expected at least one result with page content `{expected_page_content}`"
    );
}

pub async fn assert_get_by_ids_behavior<V>(
    store: &mut V,
    documents: Vec<Document>,
    requested_ids: Vec<String>,
    should_be_supported: bool,
) where
    V: VectorStoreUnderTest,
{
    store
        .add_documents(documents)
        .await
        .expect("vector store add_documents should succeed");

    let result = store.get_by_ids(requested_ids.clone()).await;

    if should_be_supported {
        let loaded = result.expect("get_by_ids should succeed");
        let loaded_ids = loaded
            .iter()
            .map(|document| document.id.as_deref())
            .collect::<Vec<_>>();
        let expected_ids = requested_ids
            .iter()
            .map(|id| Some(id.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(loaded_ids, expected_ids);
    } else {
        let error = result.expect_err("get_by_ids should fail");
        assert!(error.to_string().contains("get_by_ids"));
    }
}
