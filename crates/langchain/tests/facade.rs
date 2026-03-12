use langchain::embeddings::{CharacterEmbeddings, Embeddings};
use langchain::messages::HumanMessage;
use langchain::prompt_values::StringPromptValue;
use langchain::retrievers::VectorStoreRetriever;
use langchain::runnables::Runnable;
use langchain::text_splitters::{TextSplitter, TokenTextSplitter, Tokenizer, split_text_on_tokens};
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

#[test]
fn facade_reexports_prompt_values_and_token_helpers() {
    let prompt = StringPromptValue::new("hello");
    let tokenizer = Tokenizer::whitespace();
    let chunks = split_text_on_tokens("alpha beta gamma", &tokenizer, 2, 1);
    let splitter = TokenTextSplitter::new(Tokenizer::whitespace(), 2, 0);

    assert_eq!(prompt.to_string(), "hello");
    assert_eq!(chunks, vec!["alpha beta", "beta gamma"]);
    assert_eq!(
        splitter.split_text("alpha beta gamma"),
        vec!["alpha beta", "gamma"]
    );
}

#[tokio::test]
async fn facade_reexports_retrievers() {
    let mut store = InMemoryVectorStore::new(CharacterEmbeddings::new());
    store
        .add_documents(vec![langchain::documents::Document::new("alpha")])
        .await
        .expect("document add should succeed");

    let retriever = VectorStoreRetriever::new(store).with_limit(1);
    let documents = retriever
        .invoke("alpha".to_owned(), Default::default())
        .await
        .expect("retriever should return documents");

    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].page_content, "alpha");
}
