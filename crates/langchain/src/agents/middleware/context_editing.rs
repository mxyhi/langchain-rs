use langchain_core::messages::BaseMessage;

use crate::agents::middleware::types::{AgentMiddleware, ModelRequest};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClearToolUsesEdit {
    All,
    KeepLast(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextEditingMiddleware {
    edit: ClearToolUsesEdit,
}

impl ContextEditingMiddleware {
    pub fn new(edit: ClearToolUsesEdit) -> Self {
        Self { edit }
    }
}

impl AgentMiddleware for ContextEditingMiddleware {
    fn before_model(
        &self,
        request: &mut ModelRequest,
    ) -> Result<Option<crate::agents::middleware::types::JumpTo>, langchain_core::LangChainError>
    {
        let tool_messages = request
            .messages()
            .iter()
            .filter(|message| matches!(message, BaseMessage::Tool(_)))
            .count();
        let keep_from = match self.edit {
            ClearToolUsesEdit::All => 0,
            ClearToolUsesEdit::KeepLast(count) => tool_messages.saturating_sub(count),
        };

        let mut seen_tool_messages = 0_usize;
        request.messages_mut().retain(|message| match message {
            BaseMessage::Tool(_) => {
                let keep = seen_tool_messages >= keep_from;
                seen_tool_messages += 1;
                keep
            }
            BaseMessage::Ai(ai) if !ai.tool_calls().is_empty() => false,
            _ => true,
        });
        Ok(None)
    }
}
