use langchain_core::language_models::BaseChatModel;

#[test]
fn fireworks_namespaces_match_root_exports() {
    let root_chat = langchain_fireworks::ChatFireworks::new(
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        None::<&str>,
    );
    let namespaced_chat = langchain_fireworks::chat_models::ChatFireworks::new(
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        None::<&str>,
    );
    assert_eq!(root_chat.model_name(), namespaced_chat.model_name());

    let namespaced_llm = langchain_fireworks::llms::Fireworks::new(
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        None::<&str>,
    );
    assert_eq!(
        namespaced_llm.base_url(),
        "https://api.fireworks.ai/inference/v1"
    );

    let namespaced_embeddings = langchain_fireworks::embeddings::FireworksEmbeddings::new(
        "nomic-ai/nomic-embed-text-v1.5",
        None::<&str>,
    );
    assert_eq!(
        namespaced_embeddings.base_url(),
        "https://api.fireworks.ai/inference/v1"
    );
}
