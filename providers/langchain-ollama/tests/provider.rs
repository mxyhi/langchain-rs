use langchain_ollama::{ChatOllama, OllamaEmbeddings, OllamaLLM};

#[test]
fn ollama_types_use_local_default_base_url() {
    let chat = ChatOllama::new("llama3.1", None::<&str>);
    let llm = OllamaLLM::new("llama3.1", None::<&str>);
    let embeddings = OllamaEmbeddings::new("nomic-embed-text", None::<&str>);

    assert_eq!(chat.base_url(), "http://localhost:11434/v1");
    assert_eq!(llm.base_url(), "http://localhost:11434/v1");
    assert_eq!(embeddings.base_url(), "http://localhost:11434/v1");
}
