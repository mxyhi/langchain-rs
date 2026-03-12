use std::sync::Arc;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::messages::{ToolCall, ToolMessage};
use langchain_core::tools::BaseTool;
pub use langchain_core::tools::{InjectedState, InjectedStore, ToolRuntime};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallWithContext<State = Value> {
    tool_call: ToolCall,
    state: State,
}

impl<State> ToolCallWithContext<State> {
    pub fn new(tool_call: ToolCall, state: State) -> Self {
        Self { tool_call, state }
    }

    pub fn with_tool_call(mut self, tool_call: ToolCall) -> Self {
        self.tool_call = tool_call;
        self
    }

    pub fn with_state<NextState>(self, state: NextState) -> ToolCallWithContext<NextState> {
        ToolCallWithContext {
            tool_call: self.tool_call,
            state,
        }
    }

    pub fn tool_call(&self) -> &ToolCall {
        &self.tool_call
    }

    pub fn state(&self) -> &State {
        &self.state
    }
}

#[derive(Clone)]
pub struct ToolCallRequest<State = Value, Runtime = ToolRuntime<Value, Value>> {
    tool_call: ToolCall,
    tool: Option<Arc<dyn BaseTool>>,
    state: State,
    runtime: Runtime,
}

impl<State, Runtime> ToolCallRequest<State, Runtime> {
    pub fn new(tool_call: ToolCall, state: State, runtime: Runtime) -> Self {
        Self {
            tool_call,
            tool: None,
            state,
            runtime,
        }
    }

    pub fn with_tool(mut self, tool: Arc<dyn BaseTool>) -> Self {
        self.tool = Some(tool);
        self
    }

    pub fn with_tool_call(mut self, tool_call: ToolCall) -> Self {
        self.tool_call = tool_call;
        self
    }

    pub fn with_state<NextState>(self, state: NextState) -> ToolCallRequest<NextState, Runtime> {
        ToolCallRequest {
            tool_call: self.tool_call,
            tool: self.tool,
            state,
            runtime: self.runtime,
        }
    }

    pub fn with_runtime<NextRuntime>(
        self,
        runtime: NextRuntime,
    ) -> ToolCallRequest<State, NextRuntime> {
        ToolCallRequest {
            tool_call: self.tool_call,
            tool: self.tool,
            state: self.state,
            runtime,
        }
    }

    pub fn override_with(&self) -> Self
    where
        State: Clone,
        Runtime: Clone,
    {
        self.clone()
    }

    pub fn tool_call(&self) -> &ToolCall {
        &self.tool_call
    }

    pub fn tool(&self) -> Option<&Arc<dyn BaseTool>> {
        self.tool.as_ref()
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }
}

type ToolCallHandler<State = Value, Runtime = ToolRuntime<Value, Value>> = Arc<
    dyn Fn(
            ToolCallRequest<State, Runtime>,
        ) -> BoxFuture<'static, Result<ToolMessage, LangChainError>>
        + Send
        + Sync,
>;

pub type ToolCallWrapper<State = Value, Runtime = ToolRuntime<Value, Value>> = Arc<
    dyn Fn(
            ToolCallRequest<State, Runtime>,
            ToolCallHandler<State, Runtime>,
        ) -> BoxFuture<'static, Result<ToolMessage, LangChainError>>
        + Send
        + Sync,
>;
