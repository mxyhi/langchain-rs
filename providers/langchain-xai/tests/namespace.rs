use langchain_core::language_models::BaseChatModel;

#[test]
fn xai_chat_models_namespace_is_public() {
    let chat = langchain_xai::chat_models::ChatXAI::new("grok-2-1212", None::<&str>);
    assert_eq!(chat.model_name(), "grok-2-1212");
    assert_eq!(chat.base_url(), "https://api.x.ai/v1");

    let profile = langchain_xai::data::xai_profile();
    assert_eq!(profile.key, "xai");
    assert_eq!(profile.package_name, "langchain-xai");
    assert!(profile.capabilities.chat_model);
}
