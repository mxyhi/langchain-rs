use langchain_core::language_models::{BaseChatModel, BaseLLM};

#[test]
fn openai_namespaces_match_root_exports() {
    let root_chat =
        langchain_openai::ChatOpenAI::new("gpt-4o-mini", "https://api.test", None::<&str>);
    let namespaced_chat = langchain_openai::chat_models::ChatOpenAI::new(
        "gpt-4o-mini",
        "https://api.test",
        None::<&str>,
    );
    assert_eq!(root_chat.model_name(), namespaced_chat.model_name());
    assert_eq!(root_chat.base_url(), namespaced_chat.base_url());

    let root_embeddings = langchain_openai::OpenAIEmbeddings::new(
        "text-embedding-3-small",
        "https://api.test",
        None::<&str>,
    );
    let namespaced_embeddings = langchain_openai::embeddings::OpenAIEmbeddings::new(
        "text-embedding-3-small",
        "https://api.test",
        None::<&str>,
    );
    assert_eq!(root_embeddings.base_url(), namespaced_embeddings.base_url());

    let root_llm =
        langchain_openai::OpenAI::new("gpt-3.5-turbo-instruct", "https://api.test", None::<&str>);
    let namespaced_llm = langchain_openai::llms::OpenAI::new(
        "gpt-3.5-turbo-instruct",
        "https://api.test",
        None::<&str>,
    );
    assert_eq!(root_llm.model_name(), namespaced_llm.model_name());
    assert_eq!(root_llm.base_url(), namespaced_llm.base_url());

    let root_compatible = langchain_openai::OpenAICompatibleChatModel::new(
        "gpt-4o-mini",
        "https://compatible.test",
        None::<&str>,
    );
    let namespaced_compatible = langchain_openai::compatible::OpenAICompatibleChatModel::new(
        "gpt-4o-mini",
        "https://compatible.test",
        None::<&str>,
    );
    assert_eq!(
        root_compatible.model_name(),
        namespaced_compatible.model_name()
    );
    assert_eq!(root_compatible.base_url(), namespaced_compatible.base_url());

    let root_azure = langchain_openai::AzureChatOpenAI::new(
        "gpt-4o-mini",
        "gpt-4o-mini",
        "https://azure.test",
        None::<&str>,
    );
    let namespaced_azure = langchain_openai::azure::AzureChatOpenAI::new(
        "gpt-4o-mini",
        "gpt-4o-mini",
        "https://azure.test",
        None::<&str>,
    );
    assert_eq!(root_azure.model_name(), namespaced_azure.model_name());
    assert_eq!(root_azure.base_url(), namespaced_azure.base_url());

    let root_tool = langchain_openai::custom_tool("code_exec", "Run code");
    let namespaced_tool = langchain_openai::tools::custom_tool("code_exec", "Run code");
    assert_eq!(root_tool.name(), namespaced_tool.name());
    assert_eq!(root_tool.description(), namespaced_tool.description());

    let profile = langchain_openai::data::openai_profile();
    assert_eq!(profile.key, "openai");
    assert_eq!(profile.package_name, "langchain-openai");
    assert!(profile.capabilities.chat_model);

    let parser = langchain_openai::output_parsers::JsonOutputToolsParser::new();
    let root_parser = langchain_core::output_parsers::JsonOutputToolsParser::new();
    let _ = (parser, root_parser);

    let moderation_client =
        langchain_openai::middleware::OpenAIModerationClient::new("https://api.test", None::<&str>);
    assert_eq!(moderation_client.model(), "omni-moderation-latest");
}
