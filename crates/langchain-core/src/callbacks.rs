use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::LangChainError;
use crate::outputs::LLMResult;

static NEXT_RUN_ID: AtomicU64 = AtomicU64::new(1);

fn generate_run_id() -> String {
    format!("run-{}", NEXT_RUN_ID.fetch_add(1, Ordering::Relaxed))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallbackEventKind {
    LlmStart,
    LlmNewToken,
    LlmEnd,
    LlmError,
    ChainStart,
    ChainEnd,
    ChainError,
    ToolStart,
    ToolEnd,
    ToolError,
    Custom,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallbackEvent {
    kind: CallbackEventKind,
    run_id: String,
    parent_run_id: Option<String>,
    name: Option<String>,
    payload: Value,
    tags: Vec<String>,
    metadata: BTreeMap<String, Value>,
}

impl CallbackEvent {
    pub fn new(kind: CallbackEventKind, run: &CallbackRun, payload: impl Into<Value>) -> Self {
        Self {
            kind,
            run_id: run.id.clone(),
            parent_run_id: run.parent_run_id.clone(),
            name: run.name.clone(),
            payload: payload.into(),
            tags: run.tags.clone(),
            metadata: run.metadata.clone(),
        }
    }

    pub fn custom(name: impl Into<String>, data: impl Into<Value>, run_id: Option<&str>) -> Self {
        Self {
            kind: CallbackEventKind::Custom,
            run_id: run_id.unwrap_or("").to_owned(),
            parent_run_id: None,
            name: Some(name.into()),
            payload: data.into(),
            tags: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn kind(&self) -> CallbackEventKind {
        self.kind
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn parent_run_id(&self) -> Option<&str> {
        self.parent_run_id.as_deref()
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn payload(&self) -> &Value {
        &self.payload
    }

    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CallbackRunConfig {
    run_id: Option<String>,
    parent_run_id: Option<String>,
    name: Option<String>,
    tags: Vec<String>,
    metadata: BTreeMap<String, Value>,
}

impl CallbackRunConfig {
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    pub fn with_parent_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.parent_run_id = Some(run_id.into());
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_tags<I, S>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_metadata(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallbackRun {
    id: String,
    parent_run_id: Option<String>,
    name: Option<String>,
    tags: Vec<String>,
    metadata: BTreeMap<String, Value>,
}

impl CallbackRun {
    pub fn from_config(config: CallbackRunConfig) -> Self {
        Self {
            id: config.run_id.unwrap_or_else(generate_run_id),
            parent_run_id: config.parent_run_id,
            name: config.name,
            tags: config.tags,
            metadata: config.metadata,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn parent_run_id(&self) -> Option<&str> {
        self.parent_run_id.as_deref()
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }
}

pub trait BaseCallbackHandler: Send + Sync + 'static {
    fn as_any(&self) -> &dyn Any;

    fn on_event(&self, event: &CallbackEvent) -> Result<(), LangChainError> {
        self.dispatch_event(event)
    }

    fn on_llm_start(
        &self,
        _run: &CallbackRun,
        _serialized: &Value,
        _prompts: &[String],
    ) -> Result<(), LangChainError> {
        Ok(())
    }

    fn on_llm_new_token(&self, _run_id: &str, _token: &str) -> Result<(), LangChainError> {
        Ok(())
    }

    fn on_llm_end(&self, _run_id: &str, _response: &LLMResult) -> Result<(), LangChainError> {
        Ok(())
    }

    fn on_llm_error(&self, _run_id: &str, _error: &str) -> Result<(), LangChainError> {
        Ok(())
    }

    fn on_chain_start(
        &self,
        _run: &CallbackRun,
        _serialized: &Value,
        _inputs: &BTreeMap<String, Value>,
    ) -> Result<(), LangChainError> {
        Ok(())
    }

    fn on_chain_end(&self, _run_id: &str, _outputs: &Value) -> Result<(), LangChainError> {
        Ok(())
    }

    fn on_chain_error(&self, _run_id: &str, _error: &str) -> Result<(), LangChainError> {
        Ok(())
    }

    fn on_tool_start(
        &self,
        _run: &CallbackRun,
        _serialized: &Value,
        _input: &Value,
    ) -> Result<(), LangChainError> {
        Ok(())
    }

    fn on_tool_end(&self, _run_id: &str, _output: &Value) -> Result<(), LangChainError> {
        Ok(())
    }

    fn on_tool_error(&self, _run_id: &str, _error: &str) -> Result<(), LangChainError> {
        Ok(())
    }

    fn on_custom_event(
        &self,
        _name: &str,
        _data: &Value,
        _run_id: Option<&str>,
    ) -> Result<(), LangChainError> {
        Ok(())
    }

    fn dispatch_event(&self, event: &CallbackEvent) -> Result<(), LangChainError> {
        let run = CallbackRun {
            id: event.run_id.clone(),
            parent_run_id: event.parent_run_id.clone(),
            name: event.name.clone(),
            tags: event.tags.clone(),
            metadata: event.metadata.clone(),
        };
        match event.kind() {
            CallbackEventKind::LlmStart => {
                let serialized = event.payload().get("serialized").ok_or_else(|| {
                    LangChainError::request("llm_start event is missing `serialized`")
                })?;
                let prompts = event
                    .payload()
                    .get("prompts")
                    .and_then(Value::as_array)
                    .ok_or_else(|| LangChainError::request("llm_start event is missing `prompts`"))?
                    .iter()
                    .map(|value| {
                        value.as_str().map(str::to_owned).ok_or_else(|| {
                            LangChainError::request("llm_start prompts must be strings")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.on_llm_start(&run, serialized, &prompts)
            }
            CallbackEventKind::LlmNewToken => {
                let token = event
                    .payload()
                    .get("token")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        LangChainError::request("llm_new_token event payload is missing `token`")
                    })?;
                self.on_llm_new_token(event.run_id(), token)
            }
            CallbackEventKind::LlmEnd => {
                let response = serde_json::from_value::<LLMResult>(event.payload().clone())?;
                self.on_llm_end(event.run_id(), &response)
            }
            CallbackEventKind::LlmError => {
                let error = event
                    .payload()
                    .get("error")
                    .and_then(Value::as_str)
                    .ok_or_else(|| LangChainError::request("llm_error event is missing `error`"))?;
                self.on_llm_error(event.run_id(), error)
            }
            CallbackEventKind::ChainStart => {
                let serialized = event.payload().get("serialized").ok_or_else(|| {
                    LangChainError::request("chain_start event is missing `serialized`")
                })?;
                let inputs = event
                    .payload()
                    .get("inputs")
                    .and_then(Value::as_object)
                    .ok_or_else(|| {
                        LangChainError::request("chain_start event is missing `inputs`")
                    })?
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect::<BTreeMap<_, _>>();
                self.on_chain_start(&run, serialized, &inputs)
            }
            CallbackEventKind::ChainEnd => self.on_chain_end(event.run_id(), event.payload()),
            CallbackEventKind::ChainError => {
                let error = event
                    .payload()
                    .get("error")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        LangChainError::request("chain_error event is missing `error`")
                    })?;
                self.on_chain_error(event.run_id(), error)
            }
            CallbackEventKind::ToolStart => {
                let serialized = event.payload().get("serialized").ok_or_else(|| {
                    LangChainError::request("tool_start event is missing `serialized`")
                })?;
                let input = event.payload().get("input").ok_or_else(|| {
                    LangChainError::request("tool_start event is missing `input`")
                })?;
                self.on_tool_start(&run, serialized, input)
            }
            CallbackEventKind::ToolEnd => self.on_tool_end(event.run_id(), event.payload()),
            CallbackEventKind::ToolError => {
                let error = event
                    .payload()
                    .get("error")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        LangChainError::request("tool_error event is missing `error`")
                    })?;
                self.on_tool_error(event.run_id(), error)
            }
            CallbackEventKind::Custom => self.on_custom_event(
                event.name().unwrap_or("custom"),
                event.payload(),
                (!event.run_id().is_empty()).then_some(event.run_id()),
            ),
        }
    }
}

#[derive(Default)]
pub struct CallbackManager {
    handlers: Vec<Arc<dyn BaseCallbackHandler>>,
}

impl fmt::Debug for CallbackManager {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CallbackManager")
            .field("handler_count", &self.handlers.len())
            .finish()
    }
}

impl CallbackManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_handler<T>(mut self, handler: T) -> Self
    where
        T: BaseCallbackHandler,
    {
        self.handlers.push(Arc::new(handler));
        self
    }

    pub fn handlers(&self) -> &[Arc<dyn BaseCallbackHandler>] {
        &self.handlers
    }

    pub fn dispatch(&self, event: &CallbackEvent) -> Result<(), LangChainError> {
        for handler in &self.handlers {
            handler.on_event(event)?;
        }
        Ok(())
    }

    pub fn dispatch_custom_event(
        &self,
        name: impl Into<String>,
        data: impl Into<Value>,
        run_id: Option<&str>,
    ) -> Result<(), LangChainError> {
        self.dispatch(&CallbackEvent::custom(name, data, run_id))
    }

    pub fn on_llm_start(
        &self,
        serialized: Value,
        prompts: Vec<String>,
        config: CallbackRunConfig,
    ) -> Result<CallbackRun, LangChainError> {
        let run = CallbackRun::from_config(config);
        self.dispatch(&CallbackEvent::new(
            CallbackEventKind::LlmStart,
            &run,
            json!({
                "serialized": serialized,
                "prompts": prompts,
            }),
        ))?;
        Ok(run)
    }

    pub fn on_llm_new_token(
        &self,
        run_id: &str,
        token: impl Into<String>,
    ) -> Result<(), LangChainError> {
        let run = CallbackRun::from_config(CallbackRunConfig::default().with_run_id(run_id));
        self.dispatch(&CallbackEvent::new(
            CallbackEventKind::LlmNewToken,
            &run,
            json!({ "token": token.into() }),
        ))
    }

    pub fn on_llm_end(&self, run_id: &str, response: LLMResult) -> Result<(), LangChainError> {
        let run = CallbackRun::from_config(CallbackRunConfig::default().with_run_id(run_id));
        self.dispatch(&CallbackEvent::new(
            CallbackEventKind::LlmEnd,
            &run,
            serde_json::to_value(response)?,
        ))
    }

    pub fn on_llm_error(
        &self,
        run_id: &str,
        error: impl Into<String>,
    ) -> Result<(), LangChainError> {
        let run = CallbackRun::from_config(CallbackRunConfig::default().with_run_id(run_id));
        self.dispatch(&CallbackEvent::new(
            CallbackEventKind::LlmError,
            &run,
            json!({ "error": error.into() }),
        ))
    }

    pub fn on_chain_start(
        &self,
        serialized: Value,
        inputs: BTreeMap<String, Value>,
        config: CallbackRunConfig,
    ) -> Result<CallbackRun, LangChainError> {
        let run = CallbackRun::from_config(config);
        self.dispatch(&CallbackEvent::new(
            CallbackEventKind::ChainStart,
            &run,
            json!({
                "serialized": serialized,
                "inputs": inputs,
            }),
        ))?;
        Ok(run)
    }

    pub fn on_chain_end(&self, run_id: &str, outputs: Value) -> Result<(), LangChainError> {
        let run = CallbackRun::from_config(CallbackRunConfig::default().with_run_id(run_id));
        self.dispatch(&CallbackEvent::new(
            CallbackEventKind::ChainEnd,
            &run,
            outputs,
        ))
    }

    pub fn on_chain_error(
        &self,
        run_id: &str,
        error: impl Into<String>,
    ) -> Result<(), LangChainError> {
        let run = CallbackRun::from_config(CallbackRunConfig::default().with_run_id(run_id));
        self.dispatch(&CallbackEvent::new(
            CallbackEventKind::ChainError,
            &run,
            json!({ "error": error.into() }),
        ))
    }

    pub fn on_tool_start(
        &self,
        serialized: Value,
        input: Value,
        config: CallbackRunConfig,
    ) -> Result<CallbackRun, LangChainError> {
        let run = CallbackRun::from_config(config);
        self.dispatch(&CallbackEvent::new(
            CallbackEventKind::ToolStart,
            &run,
            json!({
                "serialized": serialized,
                "input": input,
            }),
        ))?;
        Ok(run)
    }

    pub fn on_tool_end(&self, run_id: &str, output: Value) -> Result<(), LangChainError> {
        let run = CallbackRun::from_config(CallbackRunConfig::default().with_run_id(run_id));
        self.dispatch(&CallbackEvent::new(
            CallbackEventKind::ToolEnd,
            &run,
            output,
        ))
    }

    pub fn on_tool_error(
        &self,
        run_id: &str,
        error: impl Into<String>,
    ) -> Result<(), LangChainError> {
        let run = CallbackRun::from_config(CallbackRunConfig::default().with_run_id(run_id));
        self.dispatch(&CallbackEvent::new(
            CallbackEventKind::ToolError,
            &run,
            json!({ "error": error.into() }),
        ))
    }
}
