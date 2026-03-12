use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::messages::{AIMessage, BaseMessage, HumanMessage, ToolMessage, ToolMessageStatus};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentAction {
    tool: String,
    tool_input: Value,
    log: String,
}

impl AgentAction {
    pub fn new(tool: impl Into<String>, tool_input: Value, log: impl Into<String>) -> Self {
        Self {
            tool: tool.into(),
            tool_input,
            log: log.into(),
        }
    }

    pub fn tool(&self) -> &str {
        &self.tool
    }

    pub fn tool_input(&self) -> &Value {
        &self.tool_input
    }

    pub fn log(&self) -> &str {
        &self.log
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentActionMessageLog {
    action: AgentAction,
    message_log: Vec<BaseMessage>,
}

impl AgentActionMessageLog {
    pub fn new(action: AgentAction, message_log: Vec<BaseMessage>) -> Self {
        Self {
            action,
            message_log,
        }
    }

    pub fn action(&self) -> &AgentAction {
        &self.action
    }

    pub fn message_log(&self) -> &[BaseMessage] {
        &self.message_log
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AgentActionLike {
    Action(AgentAction),
    ActionMessageLog(AgentActionMessageLog),
}

impl From<AgentAction> for AgentActionLike {
    fn from(value: AgentAction) -> Self {
        Self::Action(value)
    }
}

impl From<AgentActionMessageLog> for AgentActionLike {
    fn from(value: AgentActionMessageLog) -> Self {
        Self::ActionMessageLog(value)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentStep {
    action: AgentActionLike,
    observation: Value,
}

impl AgentStep {
    pub fn new(action: impl Into<AgentActionLike>, observation: Value) -> Self {
        Self {
            action: action.into(),
            observation,
        }
    }

    pub fn action(&self) -> &AgentActionLike {
        &self.action
    }

    pub fn observation(&self) -> &Value {
        &self.observation
    }

    pub fn messages(&self) -> Vec<BaseMessage> {
        convert_agent_observation_to_messages(&self.action, &self.observation)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentFinish {
    return_values: BTreeMap<String, Value>,
    log: String,
}

impl AgentFinish {
    pub fn new(return_values: BTreeMap<String, Value>, log: impl Into<String>) -> Self {
        Self {
            return_values,
            log: log.into(),
        }
    }

    pub fn return_values(&self) -> &BTreeMap<String, Value> {
        &self.return_values
    }

    pub fn log(&self) -> &str {
        &self.log
    }

    pub fn messages(&self) -> Vec<BaseMessage> {
        vec![BaseMessage::from(AIMessage::new(self.log.clone()))]
    }
}

pub fn convert_agent_action_to_messages(action: &AgentActionLike) -> Vec<BaseMessage> {
    match action {
        AgentActionLike::Action(action) => vec![BaseMessage::from(AIMessage::new(action.log()))],
        AgentActionLike::ActionMessageLog(action) => action.message_log().to_vec(),
    }
}

pub fn convert_agent_observation_to_messages(
    action: &AgentActionLike,
    observation: &Value,
) -> Vec<BaseMessage> {
    match action {
        AgentActionLike::Action(_action) => {
            vec![BaseMessage::from(HumanMessage::new(render_observation(
                observation,
            )))]
        }
        AgentActionLike::ActionMessageLog(action) => {
            vec![BaseMessage::from(ToolMessage::with_parts(
                render_observation(observation),
                action.action().tool(),
                Some(action.action().tool()),
                None,
                ToolMessageStatus::Success,
            ))]
        }
    }
}

fn render_observation(observation: &Value) -> String {
    match observation {
        Value::String(value) => value.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| other.to_string()),
    }
}
