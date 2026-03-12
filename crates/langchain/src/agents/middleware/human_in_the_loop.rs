use std::collections::BTreeSet;

use super::types::AgentMiddleware;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterruptOnConfig {
    All,
    None,
    ToolNames(BTreeSet<String>),
}

impl InterruptOnConfig {
    pub fn all() -> Self {
        Self::All
    }

    pub fn none() -> Self {
        Self::None
    }

    pub fn only<I, S>(tool_names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::ToolNames(tool_names.into_iter().map(Into::into).collect())
    }

    pub fn should_interrupt(&self, tool_name: &str) -> bool {
        match self {
            Self::All => true,
            Self::None => false,
            Self::ToolNames(tool_names) => tool_names.contains(tool_name),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanInterruptRequest {
    tool_name: String,
    reason: String,
}

impl HumanInterruptRequest {
    pub fn new(tool_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            reason: reason.into(),
        }
    }

    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanInTheLoopMiddleware {
    config: InterruptOnConfig,
}

impl HumanInTheLoopMiddleware {
    pub fn new(config: InterruptOnConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &InterruptOnConfig {
        &self.config
    }

    pub fn should_interrupt_tool_call(&self, tool_name: &str) -> bool {
        self.config.should_interrupt(tool_name)
    }

    pub fn build_interrupt_request(
        &self,
        tool_name: impl Into<String>,
        reason: impl Into<String>,
    ) -> Option<HumanInterruptRequest> {
        let tool_name = tool_name.into();
        self.should_interrupt_tool_call(&tool_name)
            .then(|| HumanInterruptRequest::new(tool_name, reason))
    }
}

impl AgentMiddleware for HumanInTheLoopMiddleware {}
