use langchain_core::language_models::{BaseChatModel, BaseLLM};
use langchain_core::messages::AIMessage;
use langchain_core::runnables::RunnableConfig;
use serde_json::json;

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

    let profile = langchain_anthropic::data::anthropic_profile();
    assert_eq!(profile.key, "anthropic");
    assert_eq!(profile.package_name, "langchain-anthropic");
    assert!(profile.capabilities.chat_model);
}

#[test]
fn anthropic_output_parser_and_experimental_namespaces_are_usable() {
    let message = AIMessage::new(String::new()).with_tool_calls(vec![
        langchain_core::messages::ToolCall::new("lookup", json!({ "query": "rust" }))
            .with_id("toolu_1"),
    ]);

    let parser = langchain_anthropic::output_parsers::ToolsOutputParser::new();
    let parsed = parser.parse_tool_calls(&message);
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].name(), "lookup");
    assert_eq!(parsed[0].args(), &json!({ "query": "rust" }));
    assert_eq!(
        parser.parse_first_tool_args(&message),
        Some(json!({ "query": "rust" }))
    );

    let system_message =
        langchain_anthropic::experimental::get_system_message(&[langchain_core::tools::tool(
            "lookup",
            "Look up data",
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query" }
            },
            "required": ["query"]
        }))]);
    assert!(system_message.contains("<tool_name>lookup</tool_name>"));
    assert!(system_message.contains("<name>query</name>"));

    let parsed_calls = langchain_anthropic::experimental::extract_tool_calls(
        r#"<function_calls><invoke><tool_name>lookup</tool_name><parameters><query>rust</query></parameters></invoke></function_calls>"#,
        &[langchain_core::tools::tool("lookup", "Look up data").with_parameters(json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" }
            }
        }))],
    )
    .expect("experimental parser should succeed");
    assert_eq!(parsed_calls.len(), 1);
    assert_eq!(parsed_calls[0].name(), "lookup");
    assert_eq!(parsed_calls[0].args(), &json!({ "query": "rust" }));
}

#[test]
fn anthropic_middleware_namespace_is_usable() {
    let cache = langchain_anthropic::middleware::AnthropicPromptCachingMiddleware::new()
        .with_min_messages_to_cache(2)
        .configured_config(2, RunnableConfig::default())
        .expect("cache middleware should configure request metadata");
    assert_eq!(
        cache
            .metadata
            .get(langchain_anthropic::middleware::CACHE_CONTROL_CONFIG_KEY),
        Some(&json!({ "type": "ephemeral", "ttl": "5m" }))
    );

    let tool = langchain_anthropic::middleware::ClaudeBashToolMiddleware::new(None::<&str>).tool();
    assert_eq!(tool.name(), "bash");

    let text_editor = langchain_anthropic::middleware::StateClaudeTextEditorMiddleware::new();
    assert_eq!(text_editor.tool().name(), "str_replace_based_edit_tool");

    let memory = langchain_anthropic::middleware::StateClaudeMemoryMiddleware::new();
    assert_eq!(memory.tool().name(), "memory");
    assert!(
        memory
            .system_prompt()
            .expect("memory middleware should carry system prompt")
            .contains("MEMORY PROTOCOL")
    );

    let search = langchain_anthropic::middleware::StateFileSearchMiddleware::new();
    let tools = search.tools();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].name(), "glob_search");
    assert_eq!(tools[1].name(), "grep_search");
}
