use langchain_mistralai::{ChatMistralAI, MistralAIEmbeddings};

#[test]
fn mistral_types_use_reference_default_base_url() {
    let chat = ChatMistralAI::new("mistral-small-latest", None::<&str>);
    let embeddings = MistralAIEmbeddings::new("mistral-embed", None::<&str>);

    assert_eq!(chat.base_url(), "https://api.mistral.ai/v1");
    assert_eq!(embeddings.base_url(), "https://api.mistral.ai/v1");
}
