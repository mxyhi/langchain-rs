use langchain_core::language_models::BaseChatModel;

#[test]
fn ollama_namespaces_match_root_exports() {
    let root_chat = langchain_ollama::ChatOllama::new("llama3.1", None::<&str>);
    let namespaced_chat = langchain_ollama::chat_models::ChatOllama::new("llama3.1", None::<&str>);
    assert_eq!(root_chat.model_name(), namespaced_chat.model_name());

    let namespaced_llm = langchain_ollama::llms::OllamaLLM::new("llama3.1", None::<&str>);
    assert_eq!(namespaced_llm.base_url(), "http://localhost:11434/v1");

    let namespaced_embeddings =
        langchain_ollama::embeddings::OllamaEmbeddings::new("nomic-embed-text", None::<&str>);
    assert_eq!(
        namespaced_embeddings.base_url(),
        "http://localhost:11434/v1"
    );

    let profile = langchain_ollama::data::ollama_profile();
    assert_eq!(profile.key, "ollama");
    assert_eq!(profile.package_name, "langchain-ollama");
    assert!(profile.capabilities.embeddings);
}
