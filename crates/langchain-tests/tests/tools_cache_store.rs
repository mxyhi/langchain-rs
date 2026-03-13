use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::caches::{BaseCache, InMemoryCache};
use langchain_core::messages::ToolCall;
use langchain_core::stores::InMemoryStore;
use langchain_core::tools::{BaseTool, Tool, tool};
use langchain_tests::{
    assert_async_cache_lookup, assert_cache_lookup, assert_store_roundtrip, assert_tool_invocation,
    cache_value,
};

#[derive(Clone)]
struct EchoTool {
    inner: Tool,
}

impl EchoTool {
    fn new() -> Self {
        Self {
            inner: Tool::new(tool("echo", "Echo a string"), |input| {
                Box::pin(async move { Ok(format!("echo:{input}")) })
            }),
        }
    }
}

impl BaseTool for EchoTool {
    fn definition(&self) -> &langchain_core::tools::ToolDefinition {
        self.inner.definition()
    }

    fn invoke<'a>(
        &'a self,
        input: ToolCall,
        config: langchain_core::runnables::RunnableConfig,
    ) -> BoxFuture<'a, Result<langchain_core::messages::ToolMessage, LangChainError>> {
        self.inner.invoke(input, config)
    }
}

#[tokio::test]
async fn tool_helper_validates_successful_invocation() {
    let tool = EchoTool::new();
    assert_tool_invocation(
        &tool,
        ToolCall::new("echo", serde_json::json!({ "input": "ping" })).with_id("call-1"),
        "echo:ping",
    )
    .await
    .expect("tool helper should validate tool output");
}

#[tokio::test]
async fn cache_helpers_cover_sync_and_async_lookups() {
    let cache = InMemoryCache::new();
    cache.update("prompt", "llm", cache_value("alpha"));

    assert_cache_lookup(&cache, "prompt", "llm", "alpha");
    assert_async_cache_lookup(&cache, "prompt", "llm", "alpha").await;

    cache.clear();
    assert!(cache.lookup("prompt", "llm").is_none());
}

#[tokio::test]
async fn base_store_helper_covers_sync_async_and_prefix_keys() {
    let store = InMemoryStore::<serde_json::Value>::new();
    assert_store_roundtrip(
        &store,
        vec![
            ("doc:1".to_owned(), serde_json::json!("alpha")),
            ("doc:2".to_owned(), serde_json::json!("beta")),
        ],
        Some("doc:"),
    )
    .await;
}
