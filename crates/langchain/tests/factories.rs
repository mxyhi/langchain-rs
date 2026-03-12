use langchain::embeddings::Embeddings;
use langchain::messages::HumanMessage;
use langchain::runnables::Runnable;
use langchain::{ModelInitOptions, init_chat_model, init_embeddings};
use serde_json::json;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn init_chat_model_supports_provider_prefixed_model_name() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_json(json!({
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": "ping"
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_factory",
            "model": "gpt-4o-mini",
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

    let model = init_chat_model(
        "openai:gpt-4o-mini",
        ModelInitOptions::default().with_base_url(server.uri()),
    )
    .expect("factory should create chat model");
    let message = model
        .invoke(vec![HumanMessage::new("ping").into()], Default::default())
        .await
        .expect("chat model should invoke");

    assert_eq!(message.content(), "pong");
}

#[tokio::test]
async fn init_chat_model_infers_openai_provider_from_model_prefix() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_infer",
            "model": "gpt-4o-mini",
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

    let model = init_chat_model(
        "gpt-4o-mini",
        ModelInitOptions::default().with_base_url(server.uri()),
    )
    .expect("provider inference should succeed");
    let message = model
        .invoke(vec![HumanMessage::new("ping").into()], Default::default())
        .await
        .expect("chat model should invoke");

    assert_eq!(message.content(), "pong");
}

#[tokio::test]
async fn init_embeddings_supports_explicit_provider() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "embedding": [0.1, 0.2, 0.3] }
            ]
        })))
        .mount(&server)
        .await;

    let embeddings = init_embeddings(
        "text-embedding-3-small",
        ModelInitOptions::default()
            .with_provider("openai")
            .with_base_url(server.uri()),
    )
    .expect("embedding factory should create provider instance");
    let result = embeddings
        .embed_query("hello")
        .await
        .expect("embedding request should succeed");

    assert_eq!(result, vec![0.1, 0.2, 0.3]);
}

#[test]
fn init_embeddings_requires_provider_when_model_name_cannot_be_inferred() {
    let error = match init_embeddings("text-embedding-3-small", ModelInitOptions::default()) {
        Ok(_) => panic!("missing provider should fail"),
        Err(error) => error,
    };

    assert_eq!(
        error.to_string(),
        "unsupported operation: must specify provider or use `provider:model` format for embeddings"
    );
}
