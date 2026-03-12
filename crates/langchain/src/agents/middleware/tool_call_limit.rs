use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
        let mut guard = self.seen.lock().expect("tool call limiter mutex poisoned");
        let count = guard.entry(tool_name.to_owned()).or_insert(0);
        *count += 1;
        self.thread_limit.is_none_or(|limit| *count <= limit)
            && self.run_limit.is_none_or(|limit| *count <= limit)
    }
}

impl crate::agents::middleware::types::AgentMiddleware for ToolCallLimitMiddleware {}
