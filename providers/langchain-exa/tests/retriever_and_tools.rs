use langchain_core::runnables::Runnable;
use langchain_exa::{
    ExaFindSimilarResults, ExaSearchResults, ExaSearchRetriever, SearchHit, TextContentsOptions,
};

#[tokio::test]
async fn exa_search_results_rank_and_truncate_hits() {
    let tool = ExaSearchResults::new()
        .with_hit(SearchHit::new(
            "Rust async guide",
            "https://example.com/rust",
            "Rust async runtime and futures",
            0.0,
        ))
        .with_hit(SearchHit::new(
            "SQL guide",
            "https://example.com/sql",
            "Postgres indexes and plans",
            0.0,
        ))
        .with_max_results(1)
        .with_text_options(TextContentsOptions::default().with_max_characters(12));

    let results = tool
        .invoke("rust async".to_owned(), Default::default())
        .await
        .expect("search should succeed");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].title, "Rust async guide");
    assert_eq!(results[0].text, "Rust async r");
}

#[tokio::test]
async fn exa_retriever_and_similarity_tool_return_expected_hits() {
    let retriever = ExaSearchRetriever::new().with_hit(SearchHit::new(
        "LangChain",
        "https://example.com/langchain",
        "Build retrieval pipelines in Rust",
        0.0,
    ));
    let documents = retriever
        .invoke("retrieval rust".to_owned(), Default::default())
        .await
        .expect("retriever should succeed");
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].metadata["title"], "LangChain");

    let finder = ExaFindSimilarResults::new().with_hit(SearchHit::new(
        "Agent search",
        "https://example.com/agent",
        "Agentic search workflows",
        0.0,
    ));
    let hits = finder
        .invoke("search workflows".to_owned(), Default::default())
        .await
        .expect("similarity search should succeed");
    assert_eq!(hits[0].title, "Agent search");
}

#[tokio::test]
async fn exa_root_exports_support_the_readme_quick_start_flow() {
    let search = ExaSearchResults::new()
        .with_hit(SearchHit::new(
            "Rust async guide",
            "https://example.com/rust",
            "Rust async runtime and futures",
            0.0,
        ))
        .with_text_options(TextContentsOptions::default().with_max_characters(12));
    let hits = search
        .invoke("rust".to_owned(), Default::default())
        .await
        .expect("search tool should succeed");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].title, "Rust async guide");
    assert_eq!(hits[0].text, "Rust async r");

    let similar = ExaFindSimilarResults::new().with_hit(SearchHit::new(
        "Rust async guide",
        "https://example.com/rust",
        "Rust async runtime and futures",
        0.0,
    ));
    let similar_hits = similar
        .invoke("async runtime".to_owned(), Default::default())
        .await
        .expect("similar tool should succeed");
    assert_eq!(similar_hits.len(), 1);
    assert_eq!(similar_hits[0].url, "https://example.com/rust");

    let retriever = ExaSearchRetriever::new()
        .with_max_results(1)
        .with_hit(SearchHit::new(
            "LangChain",
            "https://example.com/langchain",
            "Build retrieval pipelines in Rust",
            0.0,
        ));
    let documents = retriever
        .invoke("retrieval rust".to_owned(), Default::default())
        .await
        .expect("retriever should succeed");
    assert_eq!(documents.len(), 1);
    assert_eq!(
        documents[0].metadata["url"],
        "https://example.com/langchain"
    );
}
