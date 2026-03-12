#[test]
fn nomic_embeddings_namespace_is_public() {
    let embeddings = langchain_nomic::embeddings::NomicEmbeddings::new("nomic-embed-text-v1.5");
    assert_eq!(embeddings.model(), "nomic-embed-text-v1.5");
}
