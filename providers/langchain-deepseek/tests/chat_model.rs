use langchain_core::messages::HumanMessage;
use langchain_core::runnables::Runnable;
use langchain_deepseek::ChatDeepSeek;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn deepseek_uses_reference_default_base_url() {
    let model = ChatDeepSeek::new("deepseek-chat", None::<&str>);
    assert_eq!(model.base_url(), "https://api.deepseek.com/v1");
}

#[tokio::test]
async fn deepseek_wrapper_forwards_chat_requests() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_deepseek",
            "model": "deepseek-chat",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "pong"
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let model = ChatDeepSeek::new_with_base_url("deepseek-chat", server.uri(), Some("test-key"));
    let response = model
        .invoke(vec![HumanMessage::new("ping").into()], Default::default())
        .await
        .expect("deepseek invocation should succeed");

    assert_eq!(response.content(), "pong");
}
