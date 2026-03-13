use langchain_core::language_models::BaseLLM;
use langchain_fireworks::Fireworks;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn fireworks_wrapper_surfaces_http_status_errors() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/completions"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(503).set_body_string("provider unavailable"))
        .mount(&server)
        .await;

    let model = Fireworks::new_with_base_url(
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        server.uri(),
        Some("test-key"),
    );
    let error = model
        .generate(vec!["ping".to_owned()], Default::default())
        .await
        .expect_err("fireworks errors should surface");

    assert!(matches!(
        error,
        langchain_core::LangChainError::HttpStatus { status: 503, .. }
    ));
}
