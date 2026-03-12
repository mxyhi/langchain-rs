use wiremock::matchers::{bearer_token, body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use langchain_core::embeddings::Embeddings;
use langchain_nomic::NomicEmbeddings;

#[tokio::test]
async fn nomic_embeddings_calls_remote_embedding_api() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/embedding/text"))
        .and(bearer_token("test-key"))
        .and(body_partial_json(serde_json::json!({
            "model": "nomic-embed-text-v1.5",
            "task_type": "search_query"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embeddings": [[0.1, 0.2, 0.3]],
            "usage": { "total_tokens": 3 }
        })))
        .mount(&server)
        .await;

    let embeddings =
        NomicEmbeddings::new_with_base_url("nomic-embed-text-v1.5", server.uri(), Some("test-key"));
    assert_eq!(embeddings.model(), "nomic-embed-text-v1.5");

    let vector = embeddings
        .embed_query("ping")
        .await
        .expect("remote embedding should succeed");

    assert_eq!(vector, vec![0.1, 0.2, 0.3]);
}
