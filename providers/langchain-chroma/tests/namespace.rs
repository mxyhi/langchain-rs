#[test]
fn chroma_vectorstores_namespace_is_public() {
    let store = langchain_chroma::vectorstores::Chroma::new(
        "docs",
        langchain_core::embeddings::CharacterEmbeddings::new(),
    );
    assert_eq!(store.collection_name(), "docs");
}
