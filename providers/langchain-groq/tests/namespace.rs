use langchain_core::language_models::BaseChatModel;

#[test]
fn groq_chat_models_namespace_is_public() {
    let chat = langchain_groq::chat_models::ChatGroq::new("llama-3.1-8b-instant", None::<&str>);
    assert_eq!(chat.model_name(), "llama-3.1-8b-instant");
    assert_eq!(chat.base_url(), "https://api.groq.com/openai/v1");

    let profile = langchain_groq::data::groq_profile();
    assert_eq!(profile.key, "groq");
    assert_eq!(profile.package_name, "langchain-groq");
    assert!(profile.capabilities.chat_model);
}
