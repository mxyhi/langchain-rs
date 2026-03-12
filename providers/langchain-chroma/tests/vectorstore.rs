use langchain_chroma::Chroma;
use langchain_core::documents::Document;
use langchain_core::embeddings::CharacterEmbeddings;
use langchain_core::vectorstores::VectorStore;

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
