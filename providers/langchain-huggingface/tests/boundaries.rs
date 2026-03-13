use langchain_core::embeddings::Embeddings;
use langchain_core::language_models::{BaseChatModel, BaseLLM};
use langchain_core::messages::HumanMessage;
use langchain_huggingface::{
    ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings,
    HuggingFacePipeline,
};
use wiremock::matchers::{bearer_token, body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn huggingface_remote_boundaries_route_to_hf_inference_endpoints() {
    let chat_server = MockServer::start().await;
    let embedding_server = MockServer::start().await;
    let endpoint_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(bearer_token("hf-test"))
        .and(body_partial_json(serde_json::json!({
            "model": "meta-llama/Llama-3.1-8B-Instruct:hf-inference"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "hf-chat-1",
            "model": "meta-llama/Llama-3.1-8B-Instruct:hf-inference",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hf chat ok"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 5,
                "total_tokens": 9
            }
        })))
        .mount(&chat_server)
        .await;
    Mock::given(method("POST"))
        .and(path(
            "/hf-inference/models/sentence-transformers/all-mpnet-base-v2",
        ))
        .and(bearer_token("hf-test"))
        .and(body_partial_json(serde_json::json!({
            "inputs": ["ping"]
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!([[0.1, 0.2, 0.3]])),
        )
        .mount(&embedding_server)
        .await;
    Mock::given(method("POST"))
        .and(path("/generate"))
        .and(body_partial_json(serde_json::json!({
            "inputs": "ping"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
            {
                "generated_text": "hf endpoint ok"
            }
        ])))
        .mount(&endpoint_server)
        .await;

    let chat = ChatHuggingFace::new_with_base_url(
        "meta-llama/Llama-3.1-8B-Instruct:hf-inference",
        format!("{}/v1", chat_server.uri()),
        Some("hf-test"),
    );
    let endpoint = HuggingFaceEndpoint::new_with_base_url(
        "meta-llama/Llama-3.1-8B-Instruct",
        format!("{}/generate", endpoint_server.uri()),
        Some("hf-test"),
    )
    .with_inference_server_url("http://localhost:8010");
    let pipeline = HuggingFacePipeline::new("meta-llama/Llama-3.1-8B-Instruct");
    let embeddings = HuggingFaceEmbeddings::new_with_base_url(
        "sentence-transformers/all-mpnet-base-v2",
        embedding_server.uri(),
        Some("hf-test"),
    );
    let endpoint_embeddings = HuggingFaceEndpointEmbeddings::new_with_base_url(
        format!("{}/", embedding_server.uri()),
        Some("hf-test"),
    );

    assert_eq!(
        chat.model_name(),
        "meta-llama/Llama-3.1-8B-Instruct:hf-inference"
    );
    assert_eq!(endpoint.model_name(), "meta-llama/Llama-3.1-8B-Instruct");
    assert_eq!(pipeline.model_name(), "meta-llama/Llama-3.1-8B-Instruct");
    assert_eq!(
        endpoint
            .inference_server_url()
            .expect("inference server url should exist"),
        "http://localhost:8010"
    );
    assert_eq!(
        endpoint_embeddings.inference_server_url(),
        embedding_server.uri()
    );

    let chat_message = chat
        .generate(vec![HumanMessage::new("ping").into()], Default::default())
        .await
        .expect("chat transport should succeed");
    assert_eq!(chat_message.content(), "hf chat ok");

    let endpoint_result = endpoint
        .generate(vec!["ping".to_owned()], Default::default())
        .await
        .expect("endpoint transport should succeed");
    assert_eq!(endpoint_result.generations()[0][0].text(), "hf endpoint ok");

    let embedding = embeddings
        .embed_query("ping")
        .await
        .expect("embeddings transport should succeed");
    assert_eq!(embedding, vec![0.1, 0.2, 0.3]);

    assert!(!pipeline.is_available());
    assert_eq!(
        pipeline.unavailability_reason(),
        "HuggingFacePipeline is a boundary marker for local transformers pipelines and is not exposed as a runnable Rust BaseLLM"
    );
}
