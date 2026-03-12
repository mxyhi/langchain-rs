use langchain_core::documents::Document;
use langchain_core::embeddings::CharacterEmbeddings;
use langchain_core::vectorstores::VectorStore;
use langchain_qdrant::{FastEmbedSparse, Qdrant, RetrievalMode, SparseEmbeddings};

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
