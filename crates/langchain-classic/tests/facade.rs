use langchain_classic::chat_models::BaseChatModel;
use langchain_classic::embeddings::{CharacterEmbeddings, Embeddings};
use langchain_classic::messages::HumanMessage;
use langchain_classic::prompt_values::StringPromptValue;
use langchain_classic::retrievers::VectorStoreRetriever;
use langchain_classic::runnables::Runnable;
use langchain_classic::tools::tool;
use langchain_classic::vectorstores::{InMemoryVectorStore, VectorStore};
use langchain_core::messages::BaseMessage;

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

    let delegated = langchain_classic::chat_models::init_chat_model("openai:gpt-4o-mini")
        .expect("classic init_chat_model should delegate to the shared factory boundary");
    assert_eq!(delegated.model_name(), "gpt-4o-mini");

    let error = match langchain_classic::chat_models::init_chat_model("test-model") {
        Ok(_) => {
            panic!("model without provider should still fail when provider cannot be inferred")
        }
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("Unable to infer model provider"),
        "unexpected error: {error}"
    );

    fn assert_concrete<T>(_value: &T) {}
    assert_concrete(&langchain_classic::chat_models::ChatOpenAI::new(
        "gpt-4o-mini",
        "https://example.com",
        Option::<String>::None,
    ));
    let configurable = langchain_classic::chat_models::ConfigurableChatModel::new(
        Some("gpt-4o-mini"),
        Some("openai"),
        Some("https://api.openai.com/v1"),
        Option::<String>::None,
    );
    assert_concrete(&configurable);
}

#[tokio::test]
async fn classic_reexports_agents_boundary() {
    let agent = langchain_classic::agents::create_agent(
        langchain_classic::chat_models::ParrotChatModel::new("classic-agent", 16),
    )
    .with_system_prompt("Be concise.");

    let state = agent
        .invoke("hello".to_owned(), Default::default())
        .await
        .expect("classic agent facade should be usable");

    assert!(matches!(state.messages()[0], BaseMessage::System(_)));
    assert_eq!(state.messages()[2].content(), "hello");
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
