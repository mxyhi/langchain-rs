use langchain_fireworks::{ChatFireworks, Fireworks, FireworksEmbeddings};

#[test]
fn fireworks_types_use_reference_default_base_url() {
    let chat = ChatFireworks::new(
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        None::<&str>,
    );
    let llm = Fireworks::new(
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        None::<&str>,
    );
    let embeddings = FireworksEmbeddings::new("nomic-ai/nomic-embed-text-v1.5", None::<&str>);

    assert_eq!(chat.base_url(), "https://api.fireworks.ai/inference/v1");
    assert_eq!(llm.base_url(), "https://api.fireworks.ai/inference/v1");
    assert_eq!(
        embeddings.base_url(),
        "https://api.fireworks.ai/inference/v1"
    );
}
