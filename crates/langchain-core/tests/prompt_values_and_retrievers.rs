use langchain_core::documents::Document;
use langchain_core::embeddings::CharacterEmbeddings;
use langchain_core::messages::{BaseMessage, HumanMessage, SystemMessage};
use langchain_core::prompt_values::{ChatPromptValue, StringPromptValue};
use langchain_core::retrievers::VectorStoreRetriever;
use langchain_core::runnables::Runnable;
use langchain_core::vectorstores::{InMemoryVectorStore, VectorStore};

#[test]
fn string_prompt_value_converts_to_text_and_messages() {
    let value = StringPromptValue::new("hello");

    assert_eq!(value.to_string(), "hello");
    assert_eq!(
        value.to_messages(),
        vec![BaseMessage::from(HumanMessage::new("hello"))]
    );
}

#[test]
fn chat_prompt_value_preserves_messages_and_formats_buffer_string() {
    let value = ChatPromptValue::new(vec![
        SystemMessage::new("You are concise.").into(),
        HumanMessage::new("Summarize Rust.").into(),
    ]);

    assert_eq!(
        value.to_messages(),
        vec![
            BaseMessage::from(SystemMessage::new("You are concise.")),
            BaseMessage::from(HumanMessage::new("Summarize Rust.")),
        ]
    );
    assert_eq!(
        value.to_string(),
        "system: You are concise.\nhuman: Summarize Rust."
    );
}

#[tokio::test]
async fn vector_store_retriever_uses_similarity_search_as_runnable() {
    let mut store = InMemoryVectorStore::new(CharacterEmbeddings::new());
    store
        .add_documents(vec![Document::new("alpha"), Document::new("beta gamma")])
        .await
        .expect("documents should be added");

    let retriever = VectorStoreRetriever::new(store).with_limit(1);
    let documents = retriever
        .invoke("alpha".to_owned(), Default::default())
        .await
        .expect("retriever should return documents");

    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].page_content, "alpha");
}
