use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use futures_util::future::BoxFuture;
use langchain::agents::create_agent;
use langchain::agents::middleware::{
    ClearToolUsesEdit, CodexSandboxExecutionPolicy, ContextEditingMiddleware,
    FilesystemFileSearchMiddleware, HumanInTheLoopMiddleware, InterruptOnConfig, LLMToolEmulator,
    LLMToolSelectorMiddleware, ModelCallLimitMiddleware, ModelFallbackMiddleware,
    ModelRetryMiddleware, PIIDetectionError, PIIMiddleware, RedactionRule, ShellToolMiddleware,
    SummarizationMiddleware, TodoListMiddleware, ToolCallLimitMiddleware, ToolRetryMiddleware,
};
use langchain::language_models::{BaseChatModel, ToolBindingOptions};
use langchain::messages::{AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall};
use langchain::runnables::{Runnable, RunnableConfig};
use langchain::tools::{ToolDefinition, tool};
use serde_json::json;

#[test]
fn middleware_namespace_exposes_reference_surface() {
    let _context = ContextEditingMiddleware::new(ClearToolUsesEdit::All);
    let _retry = ModelRetryMiddleware::new().with_max_retries(3);
    let _fallback = ModelFallbackMiddleware::new(vec!["openai:gpt-4o-mini".to_owned()]);
    let _limits = ModelCallLimitMiddleware::new(Some(2), Some(1));
    let _pii = PIIMiddleware::new_email_redaction();
    let _summary = SummarizationMiddleware::new(4);
    let _todo = TodoListMiddleware::new();
    let _tool_limit = ToolCallLimitMiddleware::new(Some("search"), Some(3), Some(2));
    let _tool_retry = ToolRetryMiddleware::new().with_max_retries(2);
    let _selector = LLMToolSelectorMiddleware::allow_only(["search", "lookup"]);
    let _file_search = FilesystemFileSearchMiddleware::new(".");
    let _interrupt = HumanInTheLoopMiddleware::new(InterruptOnConfig::all());
    let _shell = ShellToolMiddleware::new(CodexSandboxExecutionPolicy::default());
    let _emulator = LLMToolEmulator::new();
    let _rule = RedactionRule::new("token", r"sk-[a-zA-Z0-9]+");
}

#[tokio::test]
async fn model_retry_retries_failed_calls_until_success() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let model = FlakyModel::new(attempts.clone(), 2, "finally works");
    let agent =
        create_agent(model).with_middleware(ModelRetryMiddleware::new().with_max_retries(3));

    let state = agent
        .invoke("hello".to_owned(), Default::default())
        .await
        .expect("retry middleware should recover");

    assert_eq!(attempts.load(Ordering::SeqCst), 3);
    assert_eq!(
        state
            .messages()
            .last()
            .expect("ai message should exist")
            .content(),
        "finally works"
    );
}

#[tokio::test]
async fn model_fallback_switches_to_secondary_model_on_primary_failure() {
    let primary_attempts = Arc::new(AtomicUsize::new(0));
    let fallback_attempts = Arc::new(AtomicUsize::new(0));
    let primary = AlwaysFailModel::new("primary-down", primary_attempts.clone());
    let fallback = EchoModel::new("fallback-ok", fallback_attempts.clone());

    let agent = create_agent(primary).with_middleware(ModelFallbackMiddleware::from_models(vec![
        Arc::new(fallback),
    ]));

    let state = agent
        .invoke("hello".to_owned(), Default::default())
        .await
        .expect("fallback should succeed");

    assert_eq!(primary_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(fallback_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(
        state
            .messages()
            .last()
            .expect("ai message should exist")
            .content(),
        "fallback-ok"
    );
}

#[tokio::test]
async fn middleware_can_redact_pii_edit_context_and_select_tools() {
    let inspector = InspectingModel::default();
    let tool_definitions = vec![
        tool("search", "search the web"),
        tool("calculator", "do math"),
    ];

    let agent = create_agent(inspector)
        .with_tools(tool_definitions, ToolBindingOptions::default())
        .with_middleware(PIIMiddleware::new_email_redaction())
        .with_middleware(ContextEditingMiddleware::new(ClearToolUsesEdit::KeepLast(
            0,
        )))
        .with_middleware(LLMToolSelectorMiddleware::allow_only(["search"]));

    let state =
        agent
            .invoke_messages(
                vec![
                    BaseMessage::from(SystemMessage::new("be useful")),
                    BaseMessage::from(HumanMessage::new("email me at jane@example.com")),
                    BaseMessage::from(AIMessage::new("tool call").with_tool_calls(vec![
                        ToolCall::new("calculator", json!({"input": "2+2"})),
                    ])),
                ],
                Default::default(),
            )
            .await
            .expect("middleware should succeed");

    let content = state
        .messages()
        .last()
        .expect("ai message should exist")
        .content()
        .to_owned();

    assert!(content.contains("[REDACTED_EMAIL]"));
    assert!(!content.contains("jane@example.com"));
    assert!(content.contains("tools=search"));
    assert!(!content.contains("calculator"));
}

#[tokio::test]
async fn summarization_and_todo_middlewares_persist_state_metadata() {
    let agent = create_agent(EchoModel::new(
        "todo:\n- write tests\n- ship middleware",
        Arc::new(AtomicUsize::new(0)),
    ))
    .with_middleware(SummarizationMiddleware::new(2))
    .with_middleware(TodoListMiddleware::new());

    let state = agent
        .invoke_messages(
            vec![
                BaseMessage::from(HumanMessage::new("first")),
                BaseMessage::from(HumanMessage::new("second")),
                BaseMessage::from(HumanMessage::new("third")),
            ],
            Default::default(),
        )
        .await
        .expect("middleware should succeed");

    assert_eq!(
        state
            .metadata()
            .get("todo.items")
            .expect("todo metadata should be captured"),
        &json!(["write tests", "ship middleware"])
    );
    assert!(
        state
            .metadata()
            .get("summaries")
            .expect("summary metadata should exist")
            .as_array()
            .expect("summaries should be an array")
            .len()
            >= 1
    );
}

#[test]
fn pii_blocking_rule_raises_detection_error() {
    let error = PIIDetectionError::new("email", "secret@example.com");
    assert!(error.to_string().contains("PII"));
}

#[derive(Clone, Default)]
struct InspectingModel {
    bound_tools: Vec<String>,
}

impl BaseChatModel for InspectingModel {
    fn model_name(&self) -> &str {
        "inspector"
    }

    fn generate<'a>(
        &'a self,
        messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, langchain::LangChainError>> {
        Box::pin(async move {
            let message_content = messages
                .iter()
                .map(BaseMessage::content)
                .collect::<Vec<_>>()
                .join(" | ");
            let tools = if self.bound_tools.is_empty() {
                "none".to_owned()
            } else {
                self.bound_tools.join(",")
            };
            Ok(AIMessage::new(format!("{message_content} | tools={tools}")))
        })
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
        _options: ToolBindingOptions,
    ) -> Result<Box<dyn BaseChatModel>, langchain::LangChainError> {
        Ok(Box::new(Self {
            bound_tools: tools.iter().map(|tool| tool.name().to_owned()).collect(),
        }))
    }
}

#[derive(Clone)]
struct FlakyModel {
    attempts: Arc<AtomicUsize>,
    failures_before_success: usize,
    response: &'static str,
}

impl FlakyModel {
    fn new(
        attempts: Arc<AtomicUsize>,
        failures_before_success: usize,
        response: &'static str,
    ) -> Self {
        Self {
            attempts,
            failures_before_success,
            response,
        }
    }
}

impl BaseChatModel for FlakyModel {
    fn model_name(&self) -> &str {
        "flaky"
    }

    fn generate<'a>(
        &'a self,
        _messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, langchain::LangChainError>> {
        Box::pin(async move {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
            if attempt <= self.failures_before_success {
                return Err(langchain::LangChainError::request(format!(
                    "temporary failure #{attempt}"
                )));
            }
            Ok(AIMessage::new(self.response))
        })
    }

    fn bind_tools(
        &self,
        _tools: Vec<ToolDefinition>,
        _options: ToolBindingOptions,
    ) -> Result<Box<dyn BaseChatModel>, langchain::LangChainError> {
        Ok(Box::new(self.clone()))
    }
}

#[derive(Clone)]
struct AlwaysFailModel {
    message: &'static str,
    attempts: Arc<AtomicUsize>,
}

impl AlwaysFailModel {
    fn new(message: &'static str, attempts: Arc<AtomicUsize>) -> Self {
        Self { message, attempts }
    }
}

impl BaseChatModel for AlwaysFailModel {
    fn model_name(&self) -> &str {
        "always-fail"
    }

    fn generate<'a>(
        &'a self,
        _messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, langchain::LangChainError>> {
        Box::pin(async move {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            Err(langchain::LangChainError::request(self.message))
        })
    }

    fn bind_tools(
        &self,
        _tools: Vec<ToolDefinition>,
        _options: ToolBindingOptions,
    ) -> Result<Box<dyn BaseChatModel>, langchain::LangChainError> {
        Ok(Box::new(self.clone()))
    }
}

#[derive(Clone)]
struct EchoModel {
    response: &'static str,
    attempts: Arc<AtomicUsize>,
}

impl EchoModel {
    fn new(response: &'static str, attempts: Arc<AtomicUsize>) -> Self {
        Self { response, attempts }
    }
}

impl BaseChatModel for EchoModel {
    fn model_name(&self) -> &str {
        "echo"
    }

    fn generate<'a>(
        &'a self,
        _messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, langchain::LangChainError>> {
        Box::pin(async move {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            Ok(AIMessage::new(self.response))
        })
    }

    fn bind_tools(
        &self,
        _tools: Vec<ToolDefinition>,
        _options: ToolBindingOptions,
    ) -> Result<Box<dyn BaseChatModel>, langchain::LangChainError> {
        Ok(Box::new(self.clone()))
    }
}
