#[test]
fn qdrant_namespaces_match_root_exports() {
    let sparse = langchain_qdrant::fastembed_sparse::FastEmbedSparse::new();
    let _trait_object: &dyn langchain_qdrant::sparse_embeddings::SparseEmbeddings = &sparse;

    let vector_store = langchain_qdrant::vectorstores::QdrantVectorStore::new(
        langchain_core::embeddings::CharacterEmbeddings::new(),
    )
    .with_retrieval_mode(langchain_qdrant::vectorstores::RetrievalMode::Hybrid);
    assert_eq!(
        vector_store.retrieval_mode(),
        langchain_qdrant::vectorstores::RetrievalMode::Hybrid
    );

    let alias_store = langchain_qdrant::qdrant::Qdrant::new(
        langchain_core::embeddings::CharacterEmbeddings::new(),
    );
    assert_eq!(
        alias_store.retrieval_mode(),
        langchain_qdrant::vectorstores::RetrievalMode::Dense
    );
}
