use langchain_core::runnables::Runnable;
use serde_json::json;

#[test]
fn perplexity_namespaces_expose_public_surface() {
    let root_chat = langchain_perplexity::ChatPerplexity::new("sonar");
    let namespaced_chat = langchain_perplexity::chat_models::ChatPerplexity::new("sonar");
    assert_eq!(root_chat.base_url(), namespaced_chat.base_url());

    let root_search_options =
        langchain_perplexity::WebSearchOptions::default().with_search_context_size(3);
    let namespaced_search_options =
        langchain_perplexity::types::WebSearchOptions::default().with_search_context_size(3);
    assert_eq!(root_search_options, namespaced_search_options);

    let json_value = langchain_perplexity::ReasoningJsonOutputParser
        .parse_str("<think>reason</think>{\"answer\":\"Boston\"}")
        .expect("root parser should succeed");
    let namespaced_json_value = langchain_perplexity::output_parsers::ReasoningJsonOutputParser
        .parse_str("<think>reason</think>{\"answer\":\"Boston\"}")
        .expect("namespaced parser should succeed");
    assert_eq!(json_value, namespaced_json_value);

    let structured = langchain_perplexity::ReasoningStructuredOutputParser
        .parse_str("<think>reason</think>{\"answer\":\"Boston\"}")
        .expect("root structured parser should succeed");
    let namespaced_structured =
        langchain_perplexity::output_parsers::ReasoningStructuredOutputParser
            .parse_str("<think>reason</think>{\"answer\":\"Boston\"}")
            .expect("namespaced structured parser should succeed");
    assert_eq!(structured, namespaced_structured);

    assert_eq!(
        langchain_perplexity::strip_think_tags("<think>reason</think>{\"answer\":\"Boston\"}"),
        langchain_perplexity::output_parsers::strip_think_tags(
            "<think>reason</think>{\"answer\":\"Boston\"}"
        )
    );
}

#[tokio::test]
async fn perplexity_namespaces_match_retriever_and_tool_exports() {
    let root_tool = langchain_perplexity::PerplexitySearchResults::new()
        .with_hit(langchain_perplexity::PerplexitySearchHit::new(
            "Rust docs",
            "https://example.com/rust",
            "Rust ownership and async patterns",
            0.0,
        ))
        .with_search_options(
            langchain_perplexity::WebSearchOptions::default().with_search_context_size(4),
        );
    let namespaced_tool = langchain_perplexity::tools::PerplexitySearchResults::new()
        .with_hit(langchain_perplexity::tools::PerplexitySearchHit::new(
            "Rust docs",
            "https://example.com/rust",
            "Rust ownership and async patterns",
            0.0,
        ))
        .with_search_options(
            langchain_perplexity::types::WebSearchOptions::default().with_search_context_size(4),
        );
    assert_eq!(
        root_tool
            .invoke("rust".to_owned(), Default::default())
            .await
            .expect("root tool should succeed"),
        namespaced_tool
            .invoke("rust".to_owned(), Default::default())
            .await
            .expect("namespaced tool should succeed")
    );

    let root_retriever = langchain_perplexity::PerplexitySearchRetriever::new().with_hit(
        langchain_perplexity::PerplexitySearchHit::new(
            "Rust docs",
            "https://example.com/rust",
            "Rust ownership and async patterns",
            0.0,
        ),
    );
    let namespaced_retriever = langchain_perplexity::retrievers::PerplexitySearchRetriever::new()
        .with_hit(langchain_perplexity::tools::PerplexitySearchHit::new(
            "Rust docs",
            "https://example.com/rust",
            "Rust ownership and async patterns",
            0.0,
        ));
    let root_documents = root_retriever
        .invoke("ownership".to_owned(), Default::default())
        .await
        .expect("root retriever should succeed");
    let namespaced_documents = namespaced_retriever
        .invoke("ownership".to_owned(), Default::default())
        .await
        .expect("namespaced retriever should succeed");
    assert_eq!(root_documents, namespaced_documents);
    assert_eq!(
        root_documents[0].metadata["url"],
        json!("https://example.com/rust")
    );
}
