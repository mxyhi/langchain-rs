use langchain_core::language_models::BaseChatModel;

#[test]
fn mistralai_namespaces_match_root_exports() {
    let chat =
        langchain_mistralai::chat_models::ChatMistralAI::new("mistral-small-latest", None::<&str>);
    assert_eq!(chat.model_name(), "mistral-small-latest");
    assert_eq!(chat.base_url(), "https://api.mistral.ai/v1");

    let embeddings =
        langchain_mistralai::embeddings::MistralAIEmbeddings::new("mistral-embed", None::<&str>);
    assert_eq!(embeddings.base_url(), "https://api.mistral.ai/v1");

    let profile = langchain_mistralai::data::mistralai_profile();
    assert_eq!(profile.key, "mistralai");
    assert_eq!(profile.package_name, "langchain-mistralai");
    assert!(profile.capabilities.embeddings);
}
