use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use langchain_core::LangChainError;
use langchain_core::messages::{AIMessage, BaseMessage};

use crate::agents::AgentState;
use crate::agents::middleware::types::{AgentMiddleware, JumpTo, ModelRequest, ModelResponse};

#[derive(Clone, Debug)]
pub struct ModelCallLimitMiddleware {
    thread_limit: Option<usize>,
    run_limit: Option<usize>,
    total_calls: Arc<AtomicUsize>,
}

impl ModelCallLimitMiddleware {
    pub fn new(thread_limit: Option<usize>, run_limit: Option<usize>) -> Self {
        Self {
            thread_limit,
            run_limit,
            total_calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl AgentMiddleware for ModelCallLimitMiddleware {
    fn before_model(&self, request: &mut ModelRequest) -> Result<Option<JumpTo>, LangChainError> {
        let current = self.total_calls.load(Ordering::SeqCst);
        let run_calls = request
            .state()
            .metadata()
            .get("model_call.run_count")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize;

        if self.thread_limit.is_some_and(|limit| current >= limit)
            || self.run_limit.is_some_and(|limit| run_calls >= limit)
        {
            request
                .messages_mut()
                .push(BaseMessage::from(AIMessage::new(
                    "Model call limit exceeded.",
                )));
            return Ok(Some(JumpTo::End));
        }

        Ok(None)
    }

    fn after_model(
        &self,
        state: &mut AgentState,
        _response: &mut ModelResponse,
    ) -> Result<Option<JumpTo>, LangChainError> {
        let total = self.total_calls.fetch_add(1, Ordering::SeqCst) + 1;
        let run_total = state
            .metadata()
            .get("model_call.run_count")
            .and_then(|value| value.as_u64())
            .unwrap_or(0)
            + 1;
        state
            .metadata_mut()
            .insert("model_call.thread_count".to_owned(), total.into());
        state
            .metadata_mut()
            .insert("model_call.run_count".to_owned(), run_total.into());
        Ok(None)
    }
}
