use langchain_core::language_models::BaseChatModel;

#[test]
fn deepseek_chat_models_namespace_is_public() {
    let chat = langchain_deepseek::chat_models::ChatDeepSeek::new("deepseek-chat", None::<&str>);
    assert_eq!(chat.model_name(), "deepseek-chat");
    assert_eq!(chat.base_url(), "https://api.deepseek.com/v1");

    let profile = langchain_deepseek::data::deepseek_profile();
    assert_eq!(profile.key, "deepseek");
    assert_eq!(profile.package_name, "langchain-deepseek");
    assert!(profile.capabilities.chat_model);
}
