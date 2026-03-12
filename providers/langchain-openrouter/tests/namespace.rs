use langchain_core::language_models::BaseChatModel;

#[test]
fn openrouter_chat_models_namespace_is_public() {
    let chat =
        langchain_openrouter::chat_models::ChatOpenRouter::new("openai/gpt-4o-mini", None::<&str>);
    assert_eq!(chat.model_name(), "openai/gpt-4o-mini");
    assert_eq!(chat.base_url(), "https://openrouter.ai/api/v1");
}
