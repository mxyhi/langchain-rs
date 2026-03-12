use langchain_openai::OpenAIEmbeddings;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn invokes_embeddings_endpoint_and_maps_vectors() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(json!({
            "model": "text-embedding-3-small",
            "input": ["alpha", "beta"],
            "dimensions": 3
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "embedding": [0.1, 0.2, 0.3] },
                { "embedding": [0.4, 0.5, 0.6] }
            ]
        })))
        .mount(&server)
        .await;

    let embeddings =
        OpenAIEmbeddings::new("text-embedding-3-small", server.uri(), Some("test-key"))
            .with_dimensions(3);

    let result = embeddings
        .embed_documents(["alpha", "beta"])
        .await
        .expect("embedding request should succeed");

    assert_eq!(result, vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]]);
}

#[tokio::test]
async fn batches_embeddings_by_chunk_size() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(body_json(json!({
            "model": "text-embedding-3-small",
            "input": ["one", "two"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "embedding": [1.0, 1.1] },
                { "embedding": [2.0, 2.1] }
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(body_json(json!({
            "model": "text-embedding-3-small",
            "input": ["three"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "embedding": [3.0, 3.1] }
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let embeddings = OpenAIEmbeddings::new("text-embedding-3-small", server.uri(), None::<&str>)
        .with_chunk_size(2);

    let result = embeddings
        .embed_documents(["one", "two", "three"])
        .await
        .expect("chunked embedding requests should succeed");

    assert_eq!(result, vec![vec![1.0, 1.1], vec![2.0, 2.1], vec![3.0, 3.1]]);
}

#[tokio::test]
async fn embed_query_returns_first_vector() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "embedding": [9.0, 9.1, 9.2] }
            ]
        })))
        .mount(&server)
        .await;

    let embeddings = OpenAIEmbeddings::new("text-embedding-3-small", server.uri(), None::<&str>);
    let result = embeddings
        .embed_query("hello")
        .await
        .expect("query embedding should succeed");

    assert_eq!(result, vec![9.0, 9.1, 9.2]);
}
