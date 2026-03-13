use langchain_core::messages::HumanMessage;
use langchain_core::runnables::Runnable;
use langchain_groq::ChatGroq;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn groq_wrapper_forwards_chat_requests() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(json!({
            "model": "llama-3.1-8b-instant",
            "messages": [
                { "role": "user", "content": "ping" }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_groq",
            "model": "llama-3.1-8b-instant",
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

    let model = ChatGroq::new_with_base_url("llama-3.1-8b-instant", server.uri(), Some("test-key"));
    let response = model
        .invoke(vec![HumanMessage::new("ping").into()], Default::default())
        .await
        .expect("groq invocation should succeed");

    assert_eq!(response.content(), "pong");
}
