use langchain_core::language_models::{BaseChatModel, BaseLLM};

#[test]
fn anthropic_namespaces_match_root_exports() {
    let root_chat = langchain_anthropic::ChatAnthropic::new(
        "claude-3-7-sonnet-latest",
        "https://api.test",
        None::<&str>,
    );
    let namespaced_chat = langchain_anthropic::chat_models::ChatAnthropic::new(
        "claude-3-7-sonnet-latest",
        "https://api.test",
        None::<&str>,
    );
    assert_eq!(root_chat.model_name(), namespaced_chat.model_name());
    assert_eq!(root_chat.base_url(), namespaced_chat.base_url());

    let root_llm = langchain_anthropic::AnthropicLLM::new(
        "claude-3-7-sonnet-latest",
        "https://api.test",
        None::<&str>,
    );
    let namespaced_llm = langchain_anthropic::llms::AnthropicLLM::new(
        "claude-3-7-sonnet-latest",
        "https://api.test",
        None::<&str>,
    );
    assert_eq!(root_llm.model_name(), namespaced_llm.model_name());

    let root_tool = langchain_anthropic::convert_to_anthropic_tool(&langchain_core::tools::tool(
        "lookup",
        "Look up data",
    ));
    let namespaced_tool = langchain_anthropic::chat_models::convert_to_anthropic_tool(
        &langchain_core::tools::tool("lookup", "Look up data"),
    );
    assert_eq!(
        serde_json::to_value(root_tool).expect("root tool should serialize"),
        serde_json::to_value(namespaced_tool).expect("namespaced tool should serialize")
    );
}
