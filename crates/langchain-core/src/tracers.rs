use std::collections::BTreeMap;
use std::fmt;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::LangChainError;
use crate::callbacks::{BaseCallbackHandler, CallbackEvent, CallbackEventKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunType {
    Llm,
    Chain,
    Tool,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Succeeded,
    Errored,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Run {
    id: String,
    parent_run_id: Option<String>,
    name: String,
    run_type: RunType,
    status: RunStatus,
    inputs: Option<Value>,
    outputs: Option<Value>,
    error: Option<String>,
    tags: Vec<String>,
    metadata: BTreeMap<String, Value>,
    events: Vec<CallbackEvent>,
}

impl Run {
    fn new(event: &CallbackEvent, run_type: RunType) -> Self {
        Self {
            id: event.run_id().to_owned(),
            parent_run_id: event.parent_run_id().map(str::to_owned),
            name: event.name().unwrap_or("run").to_owned(),
            run_type,
            status: RunStatus::Running,
            inputs: Some(event.payload().clone()),
            outputs: None,
            error: None,
            tags: event.tags().to_vec(),
            metadata: event.metadata().clone(),
            events: vec![event.clone()],
        }
    }

    fn record_event(&mut self, event: &CallbackEvent) {
        self.events.push(event.clone());
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn parent_run_id(&self) -> Option<&str> {
        self.parent_run_id.as_deref()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn run_type(&self) -> RunType {
        self.run_type
    }

    pub fn status(&self) -> RunStatus {
        self.status
    }

    pub fn outputs(&self) -> Option<&Value> {
        self.outputs.as_ref()
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }

    pub fn events(&self) -> &[CallbackEvent] {
        &self.events
    }
}

pub trait BaseTracer: BaseCallbackHandler {}

#[derive(Default)]
pub struct RunCollectorCallbackHandler {
    active_runs: Mutex<BTreeMap<String, Run>>,
    traced_runs: Mutex<Vec<Run>>,
}

impl fmt::Debug for RunCollectorCallbackHandler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunCollectorCallbackHandler")
            .field(
                "active_run_ids",
                &self
                    .active_runs
                    .lock()
                    .expect("active runs lock")
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>(),
            )
            .field(
                "traced_count",
                &self.traced_runs.lock().expect("traced runs lock").len(),
            )
            .finish()
    }
}

impl RunCollectorCallbackHandler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn traced_runs(&self) -> Vec<Run> {
        self.traced_runs.lock().expect("traced runs lock").clone()
    }

    fn start_run(&self, event: &CallbackEvent, run_type: RunType) {
        self.active_runs
            .lock()
            .expect("active runs lock")
            .insert(event.run_id().to_owned(), Run::new(event, run_type));
    }

    fn update_run(
        &self,
        run_id: &str,
        updater: impl FnOnce(&mut Run),
    ) -> Result<(), LangChainError> {
        let mut active_runs = self.active_runs.lock().expect("active runs lock");
        let run = active_runs
            .get_mut(run_id)
            .ok_or_else(|| LangChainError::unsupported(format!("run `{run_id}` is not active")))?;
        updater(run);
        Ok(())
    }

    fn finish_run(
        &self,
        run_id: &str,
        status: RunStatus,
        output: Option<Value>,
        error: Option<String>,
        event: &CallbackEvent,
    ) -> Result<(), LangChainError> {
        let mut active_runs = self.active_runs.lock().expect("active runs lock");
        let mut run = active_runs
            .remove(run_id)
            .ok_or_else(|| LangChainError::unsupported(format!("run `{run_id}` is not active")))?;
        run.status = status;
        run.outputs = output;
        run.error = error;
        run.record_event(event);
        self.traced_runs.lock().expect("traced runs lock").push(run);
        Ok(())
    }
}

impl BaseCallbackHandler for RunCollectorCallbackHandler {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn on_event(&self, event: &CallbackEvent) -> Result<(), LangChainError> {
        match event.kind() {
            CallbackEventKind::LlmStart => self.start_run(event, RunType::Llm),
            CallbackEventKind::ChainStart => self.start_run(event, RunType::Chain),
            CallbackEventKind::ToolStart => self.start_run(event, RunType::Tool),
            CallbackEventKind::LlmNewToken => {
                self.update_run(event.run_id(), |run| run.record_event(event))?;
            }
            CallbackEventKind::Custom => {
                if !event.run_id().is_empty() {
                    self.update_run(event.run_id(), |run| run.record_event(event))?;
                }
            }
            CallbackEventKind::LlmEnd => {
                return self.finish_run(
                    event.run_id(),
                    RunStatus::Succeeded,
                    Some(event.payload().clone()),
                    None,
                    event,
                );
            }
            CallbackEventKind::ChainEnd | CallbackEventKind::ToolEnd => {
                return self.finish_run(
                    event.run_id(),
                    RunStatus::Succeeded,
                    Some(event.payload().clone()),
                    None,
                    event,
                );
            }
            CallbackEventKind::LlmError
            | CallbackEventKind::ChainError
            | CallbackEventKind::ToolError => {
                let error = event
                    .payload()
                    .get("error")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                return self.finish_run(event.run_id(), RunStatus::Errored, None, error, event);
            }
        }
        Ok(())
    }
}

impl BaseTracer for RunCollectorCallbackHandler {}
