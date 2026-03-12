use langchain_core::messages::HumanMessage;
use langchain_core::runnables::Runnable;
use langchain_openai::ChatOpenAI;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn invokes_chat_completions_and_maps_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_test",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "pong"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 1,
                "total_tokens": 5
            }
        })))
        .mount(&server)
        .await;

    let model = ChatOpenAI::new("gpt-4o-mini", server.uri(), Some("test-key"));
    let message = model
        .invoke(vec![HumanMessage::new("ping").into()], Default::default())
        .await
        .expect("openai invocation should succeed");

    assert_eq!(message.content(), "pong");
    assert_eq!(
        message
            .response_metadata()
            .get("model")
            .and_then(|value| value.as_str()),
        Some("gpt-4o-mini")
    );
    assert_eq!(
        message
            .usage_metadata()
            .expect("usage metadata")
            .total_tokens,
        5
    );
}
