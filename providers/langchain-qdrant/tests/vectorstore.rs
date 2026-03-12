use langchain_core::documents::Document;
use langchain_core::embeddings::{CharacterEmbeddings, Embeddings};
use langchain_core::vectorstores::VectorStore;
use langchain_qdrant::{FastEmbedSparse, Qdrant, RetrievalMode, SparseEmbeddings};
use serde_json::json;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn qdrant_vector_store_wraps_in_memory_search() {
    let embeddings = CharacterEmbeddings::new();
    let mut store = Qdrant::new(embeddings).with_retrieval_mode(RetrievalMode::Hybrid);
    assert_eq!(store.retrieval_mode(), RetrievalMode::Hybrid);

    store
        .add_documents(vec![
            Document::new("rust async runtime"),
            Document::new("postgres replication"),
        ])
        .await
        .expect("documents should be added");

    let results = store
        .similarity_search("rust", 1)
        .await
        .expect("search should succeed");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].page_content, "rust async runtime");
}

#[tokio::test]
async fn fast_embed_sparse_returns_sparse_coordinates() {
    let embeddings = FastEmbedSparse::new();
    let vector = embeddings
        .embed_query_sparse("banana")
        .await
        .expect("sparse embedding should succeed");

    assert!(!vector.indices.is_empty());
    assert_eq!(vector.indices.len(), vector.values.len());
}

#[tokio::test(flavor = "multi_thread")]
async fn qdrant_remote_backend_upserts_queries_gets_and_deletes_points() {
    let server = MockServer::start().await;
    let embeddings = CharacterEmbeddings::new();
    let add_vector = embeddings
        .embed_query("rust async runtime")
        .await
        .expect("character embedding should succeed");
    let query_vector = embeddings
        .embed_query("rust")
        .await
        .expect("character embedding should succeed");

    Mock::given(method("PUT"))
        .and(path("/collections/docs/points"))
        .and(body_json(json!({
            "points": [{
                "id": "doc_0",
                "vector": add_vector,
                "payload": {
                    "page_content": "rust async runtime",
                    "metadata": {}
                }
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "ok"
        })))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/collections/docs/points/query"))
        .and(body_json(json!({
            "query": query_vector,
            "limit": 1,
            "with_payload": true
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": {
                "points": [{
                    "id": "doc_0",
                    "score": 0.99,
                    "payload": {
                        "page_content": "rust async runtime",
                        "metadata": {}
                    }
                }]
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/collections/docs/points"))
        .and(body_json(json!({
            "ids": ["doc_0"],
            "with_payload": true,
            "with_vector": false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": [{
                "id": "doc_0",
                "payload": {
                    "page_content": "rust async runtime",
                    "metadata": {}
                }
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/collections/docs/points/delete"))
        .and(body_json(json!({
            "points": ["doc_0"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "ok"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let mut store = Qdrant::new_remote("docs", server.uri(), embeddings)
        .with_retrieval_mode(RetrievalMode::Dense);

    let ids = store
        .add_documents(vec![Document::new("rust async runtime")])
        .await
        .expect("documents should be added to remote qdrant");
    assert_eq!(ids, vec!["doc_0"]);

    let results = store
        .similarity_search("rust", 1)
        .await
        .expect("remote search should succeed");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].page_content, "rust async runtime");

    let fetched = store
        .get_by_ids(&[String::from("doc_0")])
        .expect("remote get_by_ids should succeed");
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].id.as_deref(), Some("doc_0"));

    assert!(
        store
            .delete(&[String::from("doc_0")])
            .expect("remote delete should succeed")
    );
}
