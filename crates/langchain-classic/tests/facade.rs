use langchain_classic::chat_models::BaseChatModel;
use langchain_classic::embeddings::{CharacterEmbeddings, Embeddings};
use langchain_classic::messages::HumanMessage;
use langchain_classic::prompt_values::StringPromptValue;
use langchain_classic::retrievers::VectorStoreRetriever;
use langchain_classic::runnables::Runnable;
use langchain_classic::tools::tool;
use langchain_classic::vectorstores::{InMemoryVectorStore, VectorStore};

#[test]
fn classic_reexports_messages_tools_and_prompt_values() {
    let message = HumanMessage::new("hello");
    let definition = tool("lookup", "Look up a record");
    let prompt = StringPromptValue::new("hello");

    assert_eq!(message.content(), "hello");
    assert_eq!(definition.name(), "lookup");
    assert_eq!(prompt.to_string(), "hello");
}

#[test]
fn classic_reexports_chat_models_boundary() {
    fn assert_chat_model<T: BaseChatModel>(_model: &T) {}

    let model = langchain_classic::chat_models::ParrotChatModel::new("classic-chat-model", 12);
    assert_chat_model(&model);
    assert_eq!(model.model_name(), "classic-chat-model");

    let error = match langchain_classic::chat_models::init_chat_model("openai:gpt-4o-mini") {
        Ok(_) => {
            panic!("classic init_chat_model boundary should be explicit about unsupported state")
        }
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("classic chat_models::init_chat_model is not implemented"),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn classic_reexports_embeddings_vectorstores_and_retrievers() {
    let embeddings = CharacterEmbeddings::new();
    let vector = embeddings
        .embed_query("alpha")
        .await
        .expect("embedding should succeed");
    assert!(!vector.is_empty());

    let mut store = InMemoryVectorStore::new(CharacterEmbeddings::new());
    store
        .add_documents(vec![langchain_classic::documents::Document::new("alpha")])
        .await
        .expect("document add should succeed");

    let retriever = VectorStoreRetriever::new(store).with_limit(1);
    let documents = retriever
        .invoke("alpha".to_owned(), Default::default())
        .await
        .expect("retriever should return documents");

    assert_eq!(documents[0].page_content, "alpha");
}
