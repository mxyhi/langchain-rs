use std::sync::Arc;

use langchain::chat_models::BaseChatModel;
use langchain::embeddings::{CharacterEmbeddings, Embeddings};
use langchain::messages::{HumanMessage, ToolCall};
use langchain::prompt_values::StringPromptValue;
use langchain::retrievers::VectorStoreRetriever;
use langchain::runnables::Runnable;
use langchain::text_splitters::{TextSplitter, TokenTextSplitter, Tokenizer, split_text_on_tokens};
use langchain::tools::tool_node::{
    InjectedState as ToolNodeInjectedState, InjectedStore as ToolNodeInjectedStore,
    ToolCallRequest, ToolCallWithContext, ToolCallWrapper, ToolRuntime as ToolNodeRuntime,
};
use langchain::tools::{BaseTool, Tool, ToolDefinition, tool};
use langchain::vectorstores::{InMemoryVectorStore, VectorStore};
use serde_json::{Value, json};

#[test]
fn facade_reexports_messages() {
    let message = HumanMessage::new("hello");
    assert_eq!(message.content(), "hello");
}

#[test]
fn facade_reexports_base_chat_model_from_chat_models_module() {
    fn assert_chat_model<T: BaseChatModel>(_model: &T) {}

    let model = langchain::chat_models::ParrotChatModel::new("facade-chat-model", 16);
    assert_chat_model(&model);
    assert_eq!(model.model_name(), "facade-chat-model");
}

#[test]
fn facade_reexports_tool_helper() {
    let definition = tool("lookup", "Look up a record");

    assert_eq!(definition.name(), "lookup");
    assert_eq!(definition.description(), "Look up a record");
}

#[test]
fn facade_reexports_tool_node_compat_surface() {
    let _: ToolNodeInjectedState<Value> = Default::default();
    let _: ToolNodeInjectedStore<Value> = Default::default();

    let runtime = ToolNodeRuntime::new(json!({"messages": []}), json!({"writes": []}))
        .with_tool_call_id("call_lookup_1");
    let tool_call = ToolCall::new("lookup", json!({"input": "rust"})).with_id("call_lookup_1");
    let tool: Arc<dyn BaseTool> = Arc::new(Tool::new(
        ToolDefinition::new("lookup", "Look up a record"),
        |_input| Box::pin(async move { Ok("done".to_owned()) }),
    ));

    let request = ToolCallRequest::new(
        tool_call.clone(),
        json!({"messages": ["hello"]}),
        runtime.clone(),
    )
    .with_tool(tool.clone());
    let updated = request.override_with().with_tool_call(
        ToolCall::new("lookup", json!({"input": "langchain"})).with_id("call_lookup_1"),
    );

    assert_eq!(request.tool_call().args()["input"], "rust");
    assert_eq!(updated.tool_call().args()["input"], "langchain");
    assert_eq!(request.state()["messages"][0], "hello");
    assert_eq!(request.runtime().tool_call_id(), Some("call_lookup_1"));
    assert!(Arc::ptr_eq(
        request.tool().expect("tool should be attached"),
        &tool,
    ));

    let with_context = ToolCallWithContext::new(tool_call, json!({"messages": ["hello"]}));
    assert_eq!(with_context.tool_call().name(), "lookup");
    assert_eq!(with_context.state()["messages"][0], "hello");

    let wrapper: ToolCallWrapper = Arc::new(|request, handler| handler(request));
    let _ = wrapper;
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
