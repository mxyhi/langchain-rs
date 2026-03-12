use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::documents::Document;
use langchain_tests::{RetrieverUnderTest, assert_retriever_returns_expected_document};

struct FakeRetriever {
    documents: Vec<Document>,
}

impl RetrieverUnderTest for FakeRetriever {
    fn invoke<'a>(
        &'a self,
        query: &'a str,
    ) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        Box::pin(async move {
            Ok(self
                .documents
                .iter()
                .filter(|document| document.page_content.contains(query))
                .cloned()
                .collect())
        })
    }
}

#[tokio::test]
async fn retriever_helper_finds_expected_document() {
    let retriever = FakeRetriever {
        documents: vec![Document::new("alpha"), Document::new("beta")],
    };

    assert_retriever_returns_expected_document(&retriever, "alp", "alpha").await;
}
