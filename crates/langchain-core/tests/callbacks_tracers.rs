use std::collections::BTreeMap;
use std::sync::Mutex;

use langchain_core::callbacks::{
    BaseCallbackHandler, CallbackEvent, CallbackEventKind, CallbackManager, CallbackRunConfig,
};
use langchain_core::outputs::{Generation, LLMResult};
use langchain_core::tracers::{BaseTracer, RunCollectorCallbackHandler, RunStatus, RunType};
use serde_json::{Value, json};

#[derive(Default)]
struct RecordingHandler {
    events: Mutex<Vec<CallbackEventKind>>,
    tokens: Mutex<Vec<String>>,
    custom_payloads: Mutex<Vec<Value>>,
}

impl RecordingHandler {
    fn events(&self) -> Vec<CallbackEventKind> {
        self.events.lock().expect("events lock").clone()
    }

    fn tokens(&self) -> Vec<String> {
        self.tokens.lock().expect("tokens lock").clone()
    }

    fn custom_payloads(&self) -> Vec<Value> {
        self.custom_payloads
            .lock()
            .expect("custom payload lock")
            .clone()
    }
}

impl BaseCallbackHandler for RecordingHandler {
    fn as_any(&self) -> &(dyn std::any::Any + 'static) {
        self
    }

    fn on_event(&self, event: &CallbackEvent) -> Result<(), langchain_core::LangChainError> {
        self.events.lock().expect("events lock").push(event.kind());
        self.dispatch_event(event)
    }

    fn on_llm_new_token(
        &self,
        _run_id: &str,
        token: &str,
    ) -> Result<(), langchain_core::LangChainError> {
        self.tokens
            .lock()
            .expect("tokens lock")
            .push(token.to_owned());
        Ok(())
    }

    fn on_custom_event(
        &self,
        _name: &str,
        data: &Value,
        _run_id: Option<&str>,
    ) -> Result<(), langchain_core::LangChainError> {
        self.custom_payloads
            .lock()
            .expect("custom payload lock")
            .push(data.clone());
        Ok(())
    }
}

#[test]
fn callback_manager_dispatches_specialized_events() {
    let recorder = RecordingHandler::default();
    let manager = CallbackManager::new().with_handler(recorder);

    let run = manager
        .on_llm_start(
            json!({"name": "parrot"}),
            vec!["hello".to_owned()],
            CallbackRunConfig::default().with_name("llm-call"),
        )
        .expect("llm start should dispatch");
    manager
        .on_llm_new_token(run.id(), "he")
        .expect("token dispatch should succeed");
    manager
        .dispatch_custom_event("progress", json!({"step": 1}), Some(run.id()))
        .expect("custom event dispatch should succeed");
    manager
        .on_llm_end(
            run.id(),
            LLMResult::new(vec![vec![Generation::new("hello")]]),
        )
        .expect("llm end should dispatch");

    let handler = manager.handlers()[0]
        .as_any()
        .downcast_ref::<RecordingHandler>()
        .expect("handler should downcast");
    assert_eq!(
        handler.events(),
        vec![
            CallbackEventKind::LlmStart,
            CallbackEventKind::LlmNewToken,
            CallbackEventKind::Custom,
            CallbackEventKind::LlmEnd,
        ]
    );
    assert_eq!(handler.tokens(), vec!["he".to_owned()]);
    assert_eq!(handler.custom_payloads(), vec![json!({"step": 1})]);
}

#[test]
fn run_collector_tracks_nested_runs_and_errors() {
    let collector = RunCollectorCallbackHandler::new();
    let manager = CallbackManager::new().with_handler(collector);

    let chain = manager
        .on_chain_start(
            json!({"name": "qa-chain"}),
            BTreeMap::from([("question".to_owned(), json!("where"))]),
            CallbackRunConfig::default().with_name("chain"),
        )
        .expect("chain start should succeed");
    let tool = manager
        .on_tool_start(
            json!({"name": "lookup"}),
            json!({"query": "rust"}),
            CallbackRunConfig::default()
                .with_name("lookup")
                .with_parent_run_id(chain.id()),
        )
        .expect("tool start should succeed");
    manager
        .on_tool_end(tool.id(), json!({"answer": "Rust"}))
        .expect("tool end should succeed");
    manager
        .on_chain_error(chain.id(), "failed to synthesize")
        .expect("chain error should succeed");

    let collector = manager.handlers()[0]
        .as_any()
        .downcast_ref::<RunCollectorCallbackHandler>()
        .expect("collector should downcast");
    let traced = collector.traced_runs();

    assert_eq!(traced.len(), 2);
    let tool_run = traced
        .iter()
        .find(|run: &&langchain_core::tracers::Run| run.run_type() == RunType::Tool)
        .expect("tool run should exist");
    assert_eq!(tool_run.parent_run_id(), Some(chain.id()));
    assert_eq!(tool_run.status(), RunStatus::Succeeded);
    assert_eq!(tool_run.outputs(), Some(&json!({"answer": "Rust"})));

    let chain_run = traced
        .iter()
        .find(|run: &&langchain_core::tracers::Run| run.run_type() == RunType::Chain)
        .expect("chain run should exist");
    assert_eq!(chain_run.status(), RunStatus::Errored);
    assert_eq!(chain_run.error(), Some("failed to synthesize"));
}

#[test]
fn run_collector_keeps_llm_event_log() {
    fn assert_base_tracer<T: BaseTracer>(_tracer: &T) {}

    let collector = RunCollectorCallbackHandler::new();
    assert_base_tracer(&collector);

    let manager = CallbackManager::new().with_handler(collector);
    let run = manager
        .on_llm_start(
            json!({"name": "streamer"}),
            vec!["hello".to_owned()],
            CallbackRunConfig::default()
                .with_name("streamer")
                .with_tags(vec!["demo"])
                .with_metadata(BTreeMap::from([("source".to_owned(), json!("test"))])),
        )
        .expect("llm start should succeed");
    manager
        .on_llm_new_token(run.id(), "hel")
        .expect("first token should succeed");
    manager
        .on_llm_new_token(run.id(), "lo")
        .expect("second token should succeed");
    manager
        .on_llm_end(
            run.id(),
            LLMResult::new(vec![vec![Generation::new("hello")]])
                .with_output(BTreeMap::from([("model".to_owned(), json!("parrot"))])),
        )
        .expect("llm end should succeed");

    let collector = manager.handlers()[0]
        .as_any()
        .downcast_ref::<RunCollectorCallbackHandler>()
        .expect("collector should downcast");
    let traced = collector.traced_runs();
    let run = traced.first().expect("one llm run should be collected");

    assert_eq!(run.name(), "streamer");
    assert_eq!(run.tags(), &["demo".to_owned()]);
    assert_eq!(run.metadata()["source"], "test");
    assert_eq!(run.status(), RunStatus::Succeeded);
    assert_eq!(
        run.outputs()
            .and_then(Value::as_object)
            .and_then(|value| value.get("llm_output"))
            .and_then(Value::as_object)
            .and_then(|value| value.get("model")),
        Some(&json!("parrot"))
    );
    assert_eq!(
        run.events()
            .iter()
            .map(|event: &CallbackEvent| event.kind())
            .collect::<Vec<_>>(),
        vec![
            CallbackEventKind::LlmStart,
            CallbackEventKind::LlmNewToken,
            CallbackEventKind::LlmNewToken,
            CallbackEventKind::LlmEnd,
        ]
    );
}
