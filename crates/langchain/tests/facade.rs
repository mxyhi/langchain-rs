use langchain::embeddings::{CharacterEmbeddings, Embeddings};
use langchain::messages::HumanMessage;
use langchain::tools::tool;
use langchain::vectorstores::{InMemoryVectorStore, VectorStore};

#[test]
fn facade_reexports_messages() {
    let message = HumanMessage::new("hello");
    assert_eq!(message.content(), "hello");
}

#[test]
fn facade_reexports_tool_helper() {
    let definition = tool("lookup", "Look up a record");

    assert_eq!(definition.name(), "lookup");
    assert_eq!(definition.description(), "Look up a record");
}

#[tokio::test]
async fn facade_reexports_embeddings_and_vectorstores() {
    let embeddings = CharacterEmbeddings::new();
    let vector = embeddings
        .embed_query("alpha")
        .await
        .expect("embedding should succeed");

    assert!(!vector.is_empty());

    let mut store = InMemoryVectorStore::new(CharacterEmbeddings::new());
    let ids = store
        .add_documents(vec![langchain::documents::Document::new("alpha")])
        .await
        .expect("document add should succeed");

    assert_eq!(ids.len(), 1);
}
