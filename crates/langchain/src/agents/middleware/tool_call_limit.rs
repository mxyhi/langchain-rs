use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;

use crate::agents::middleware::types::{AgentMiddleware, ToolCallHandler, ToolCallRequest};

#[derive(Clone, Debug)]
pub struct ToolCallLimitMiddleware {
    tool_name: Option<String>,
    thread_limit: Option<usize>,
    run_limit: Option<usize>,
    seen: Arc<Mutex<HashMap<String, usize>>>,
}

impl ToolCallLimitMiddleware {
    pub fn new(
        tool_name: Option<&str>,
        thread_limit: Option<usize>,
        run_limit: Option<usize>,
    ) -> Self {
        Self {
            tool_name: tool_name.map(ToOwned::to_owned),
            thread_limit,
            run_limit,
            seen: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn name(&self) -> String {
        match &self.tool_name {
            Some(name) => format!("ToolCallLimitMiddleware[{name}]"),
            None => "ToolCallLimitMiddleware".to_owned(),
        }
    }

    pub fn record(&self, tool_name: &str) -> bool {
        if self
            .tool_name
            .as_deref()
            .is_some_and(|configured| configured != tool_name)
        {
            return true;
        }

        let mut guard = self.seen.lock().expect("tool call limiter mutex poisoned");
        // Until the Rust agent grows per-run tool execution state, the limiter can only
        // observe actual tool invocations. A global limiter therefore tracks "__all__",
        // while a tool-specific limiter tracks that single configured tool.
        let key = self.tool_name.as_deref().unwrap_or("__all__");
        let count = guard.entry(key.to_owned()).or_insert(0);
        *count += 1;
        let allowed = self.thread_limit.is_none_or(|limit| *count <= limit)
            && self.run_limit.is_none_or(|limit| *count <= limit);
        if !allowed {
            *count = count.saturating_sub(1);
        }
        allowed
    }
}

impl AgentMiddleware for ToolCallLimitMiddleware {
    fn wrap_tool_call(
        &self,
        request: ToolCallRequest,
        handler: ToolCallHandler,
    ) -> BoxFuture<'static, Result<langchain_core::messages::ToolMessage, LangChainError>> {
        let allowed = self.record(request.tool_call().name());
        let tool_name = request.tool_call().name().to_owned();
        Box::pin(async move {
            if !allowed {
                return Err(LangChainError::request(format!(
                    "tool call limit exceeded for `{tool_name}`"
                )));
            }
            handler(request).await
        })
    }
}
