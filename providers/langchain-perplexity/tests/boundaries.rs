use langchain_core::runnables::Runnable;
use langchain_perplexity::{
    ChatPerplexity, PerplexitySearchHit, PerplexitySearchResults, PerplexitySearchRetriever,
    ReasoningJsonOutputParser, ReasoningStructuredOutputParser, WebSearchOptions, strip_think_tags,
};

#[test]
fn think_tags_are_removed_before_json_parsing() {
    let content = "<think>chain of thought</think>{\"answer\":\"Boston\"}";
    assert_eq!(strip_think_tags(content), "{\"answer\":\"Boston\"}");
}

#[tokio::test]
async fn reasoning_parsers_and_search_boundaries_work() {
    let json_parser = ReasoningJsonOutputParser;
    let parsed = json_parser
        .invoke(
            "<think>reason</think>{\"answer\":\"Boston\"}".to_owned(),
            Default::default(),
        )
        .await
        .expect("json parser should succeed");
    assert_eq!(parsed["answer"], "Boston");

    let structured = ReasoningStructuredOutputParser
        .invoke(
            "<think>reason</think>{\"answer\":\"Boston\"}".to_owned(),
            Default::default(),
        )
        .await
        .expect("structured parser should succeed");
    assert_eq!(structured.reasoning.as_deref(), Some("reason"));
    assert_eq!(structured.value["answer"], "Boston");

    let tool = PerplexitySearchResults::new()
        .with_hit(PerplexitySearchHit::new(
            "Rust docs",
            "https://example.com/rust",
            "Rust ownership and async patterns",
            0.0,
        ))
        .with_search_options(WebSearchOptions::default().with_search_context_size(4));
    let hits = tool
        .invoke("rust async".to_owned(), Default::default())
        .await
        .expect("search tool should succeed");
    assert_eq!(hits[0].title, "Rust docs");

    let retriever = PerplexitySearchRetriever::new().with_hit(PerplexitySearchHit::new(
        "Rust docs",
        "https://example.com/rust",
        "Rust ownership and async patterns",
        0.0,
    ));
    let documents = retriever
        .invoke("ownership".to_owned(), Default::default())
        .await
        .expect("retriever should succeed");
    assert_eq!(documents[0].metadata["url"], "https://example.com/rust");
}

#[tokio::test]
async fn chat_perplexity_is_explicitly_unsupported_for_now() {
    use langchain_core::language_models::BaseChatModel;

    let error = ChatPerplexity::new("sonar")
        .generate(Vec::new(), Default::default())
        .await
        .expect_err("chat transport should be explicit unsupported");

    assert!(
        error
            .to_string()
            .contains("ChatPerplexity transport is not implemented")
    );
}
