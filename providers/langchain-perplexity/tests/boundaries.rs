use langchain_core::runnables::Runnable;
use langchain_perplexity::{
    ChatPerplexity, PerplexitySearchHit, PerplexitySearchResults, PerplexitySearchRetriever,
    ReasoningJsonOutputParser, ReasoningStructuredOutputParser, WebSearchOptions, strip_think_tags,
};
use serde_json::json;
use wiremock::matchers::{bearer_token, body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

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
async fn chat_perplexity_uses_official_sonar_api() {
    use langchain_core::language_models::BaseChatModel;
    use langchain_core::messages::HumanMessage;

    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/sonar"))
        .and(bearer_token("test-key"))
        .and(body_partial_json(json!({
            "model": "sonar",
            "web_search_options": {
                "search_context_size": 3
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "pplx-chat-1",
            "model": "sonar",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "perplexity ok"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12
            },
            "citations": ["https://www.rust-lang.org/"]
        })))
        .mount(&server)
        .await;

    let message = ChatPerplexity::new_with_base_url("sonar", server.uri(), Some("test-key"))
        .with_web_search_options(WebSearchOptions::default().with_search_context_size(3))
        .generate(vec![HumanMessage::new("ping").into()], Default::default())
        .await
        .expect("chat request should succeed");

    assert_eq!(message.content(), "perplexity ok");
    assert_eq!(
        message.response_metadata()["citations"],
        json!(["https://www.rust-lang.org/"])
    );
}
