use std::collections::BTreeSet;

use langchain_core::LangChainError;

use crate::agents::middleware::types::{AgentMiddleware, ModelRequest};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LLMToolSelectorMiddleware {
    allowlist: BTreeSet<String>,
}

impl LLMToolSelectorMiddleware {
    pub fn allow_only(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            allowlist: names.into_iter().map(Into::into).collect(),
        }
    }
}

impl AgentMiddleware for LLMToolSelectorMiddleware {
    fn before_model(
        &self,
        request: &mut ModelRequest,
    ) -> Result<Option<crate::agents::middleware::types::JumpTo>, LangChainError> {
        request
            .tools_mut()
            .retain(|tool| self.allowlist.contains(tool.name()));
        Ok(None)
    }
}
