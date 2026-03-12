use langchain_chroma::Chroma;
use langchain_core::documents::Document;
use langchain_core::embeddings::{CharacterEmbeddings, Embeddings};
use langchain_core::vectorstores::VectorStore;
use serde_json::json;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn chroma_exposes_collection_name_and_search() {
    let embeddings = CharacterEmbeddings::new();
    let mut store = Chroma::new("docs", embeddings);
    assert_eq!(store.collection_name(), "docs");

    store
        .add_documents(vec![Document::new("langchain rust port")])
        .await
        .expect("document should be added");

    let results = store
        .similarity_search("rust", 1)
        .await
        .expect("search should succeed");
    assert_eq!(results[0].page_content, "langchain rust port");
}

#[tokio::test(flavor = "multi_thread")]
async fn chroma_remote_backend_adds_queries_gets_and_deletes_records() {
    let server = MockServer::start().await;
    let embeddings = CharacterEmbeddings::new();
    let add_vector = embeddings
        .embed_query("langchain rust port")
        .await
        .expect("character embedding should succeed");
    let query_vector = embeddings
        .embed_query("rust")
        .await
        .expect("character embedding should succeed");

    Mock::given(method("POST"))
        .and(path(
            "/api/v2/tenants/default_tenant/databases/default_database/collections",
        ))
        .and(body_json(json!({
            "name": "docs",
            "get_or_create": true
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "collection-123",
            "name": "docs"
        })))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path(
            "/api/v2/tenants/default_tenant/databases/default_database/collections/collection-123/add",
        ))
        .and(body_json(json!({
            "ids": ["doc_0"],
            "documents": ["langchain rust port"],
            "metadatas": [{}],
            "embeddings": [add_vector]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ids": ["doc_0"]
        })))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path(
            "/api/v2/tenants/default_tenant/databases/default_database/collections/collection-123/query",
        ))
        .and(body_json(json!({
            "query_embeddings": [query_vector],
            "n_results": 1,
            "include": ["documents", "metadatas", "distances"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "documents": [["langchain rust port"]],
            "metadatas": [[{}]],
            "ids": [["doc_0"]],
            "distances": [[0.01]]
        })))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path(
            "/api/v2/tenants/default_tenant/databases/default_database/collections/collection-123/get",
        ))
        .and(body_json(json!({
            "ids": ["doc_0"],
            "include": ["documents", "metadatas"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "documents": ["langchain rust port"],
            "metadatas": [{}],
            "ids": ["doc_0"]
        })))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path(
            "/api/v2/tenants/default_tenant/databases/default_database/collections/collection-123/delete",
        ))
        .and(body_json(json!({
            "ids": ["doc_0"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ids": ["doc_0"]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let mut store = Chroma::new_remote("docs", server.uri(), embeddings);

    let ids = store
        .add_documents(vec![Document::new("langchain rust port")])
        .await
        .expect("document should be added to remote chroma");
    assert_eq!(ids, vec!["doc_0"]);

    let results = store
        .similarity_search("rust", 1)
        .await
        .expect("remote search should succeed");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].page_content, "langchain rust port");
    assert_eq!(results[0].id.as_deref(), Some("doc_0"));

    let fetched = store
        .get_by_ids(&[String::from("doc_0")])
        .expect("remote get_by_ids should succeed");
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].page_content, "langchain rust port");

    assert!(
        store
            .delete(&[String::from("doc_0")])
            .expect("remote delete should succeed")
    );
}
