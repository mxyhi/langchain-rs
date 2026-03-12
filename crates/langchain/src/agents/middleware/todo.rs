use crate::agents::AgentState;
use crate::agents::middleware::types::{AgentMiddleware, ModelResponse};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TodoListMiddleware;

impl TodoListMiddleware {
    pub fn new() -> Self {
        Self
    }
}

impl AgentMiddleware for TodoListMiddleware {
    fn after_model(
        &self,
        state: &mut AgentState,
        response: &mut ModelResponse,
    ) -> Result<Option<crate::agents::middleware::types::JumpTo>, langchain_core::LangChainError>
    {
        let todos = response
            .result()
            .iter()
            .flat_map(|message| message.content().lines())
            .map(str::trim)
            .filter_map(|line| {
                line.strip_prefix("- ")
                    .or_else(|| line.strip_prefix("* "))
                    .map(str::trim)
            })
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();

        if !todos.is_empty() {
            state
                .metadata_mut()
                .insert("todo.items".to_owned(), serde_json::json!(todos));
        }
        Ok(None)
    }
}
