use langchain_core::LangChainError;

use crate::agents::middleware::types::{AgentMiddleware, ModelRequest};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummarizationMiddleware {
    max_messages: usize,
}

impl SummarizationMiddleware {
    pub fn new(max_messages: usize) -> Self {
        Self { max_messages }
    }
}

impl AgentMiddleware for SummarizationMiddleware {
    fn before_model(
        &self,
        request: &mut ModelRequest,
    ) -> Result<Option<crate::agents::middleware::types::JumpTo>, LangChainError> {
        if request.messages().len() <= self.max_messages {
            return Ok(None);
        }

        let split_at = request.messages().len().saturating_sub(self.max_messages);
        let summary = request.messages()[..split_at]
            .iter()
            .map(|message| message.content())
            .collect::<Vec<_>>()
            .join(" | ");
        request
            .state_mut()
            .metadata_mut()
            .entry("summaries".to_owned())
            .and_modify(|value| {
                if let Some(existing) = value.as_array_mut() {
                    existing.push(summary.clone().into());
                }
            })
            .or_insert_with(|| serde_json::json!([summary.clone()]));
        *request.messages_mut() = request.messages()[split_at..].to_vec();
        Ok(None)
    }
}
