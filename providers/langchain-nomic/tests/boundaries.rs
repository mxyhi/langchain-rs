use langchain_core::embeddings::Embeddings;
use langchain_nomic::NomicEmbeddings;

#[tokio::test]
async fn nomic_embeddings_is_a_real_boundary_type() {
    let embeddings = NomicEmbeddings::new("nomic-embed-text-v1.5");
    assert_eq!(embeddings.model(), "nomic-embed-text-v1.5");
    assert!(embeddings
        .embed_query("ping")
        .await
        .expect_err("transport should be unsupported")
        .to_string()
        .contains("not implemented"));
}
