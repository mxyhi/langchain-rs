use futures_util::future::BoxFuture;

use langchain_core::LangChainError;
use langchain_core::documents::Document;

pub trait RetrieverUnderTest: Send + Sync {
    fn invoke<'a>(&'a self, query: &'a str)
    -> BoxFuture<'a, Result<Vec<Document>, LangChainError>>;
}

pub async fn assert_retriever_returns_expected_document<R>(
    retriever: &R,
    query: &str,
    expected_page_content: &str,
) where
    R: RetrieverUnderTest,
{
    let documents = retriever
        .invoke(query)
        .await
        .expect("retriever should return documents");

    assert!(
        documents
            .iter()
            .any(|document| document.page_content == expected_page_content),
        "expected at least one document with page content `{expected_page_content}`"
    );
}
