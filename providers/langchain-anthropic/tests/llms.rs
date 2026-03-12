use langchain_anthropic::AnthropicLLM;
use langchain_core::language_models::BaseLLM;
use langchain_core::runnables::Runnable;
use serde_json::json;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn anthropic_llm_uses_messages_api_for_text_prompts() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": "claude-3-7-sonnet-latest",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": "alpha"
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_llm_1",
            "model": "claude-3-7-sonnet-latest",
            "content": [
                { "type": "text", "text": "ALPHA" }
            ],
            "usage": {
                "input_tokens": 5,
                "output_tokens": 5
            }
        })))
        .mount(&server)
        .await;

    let model = AnthropicLLM::new("claude-3-7-sonnet-latest", server.uri(), None::<&str>);
    let result = model
        .generate(vec!["alpha".to_owned()], Default::default())
        .await
        .expect("llm generation should succeed");
    let output = model
        .invoke("alpha".to_owned(), Default::default())
        .await
        .expect("llm invoke should succeed");

    assert_eq!(result.generations()[0][0].text(), "ALPHA");
    assert_eq!(output, "ALPHA");
}
