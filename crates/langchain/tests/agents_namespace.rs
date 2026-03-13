use std::sync::Arc;

use langchain::agents::create_agent;
use langchain::agents::middleware::types::{
    AgentMiddleware, AgentState, ExtendedModelResponse, JumpTo, ModelRequest, ModelResponse,
    ToolCallRequest, ToolCallWrapper,
};
use langchain::agents::structured_output::{
    AutoStrategy, MultipleStructuredOutputsError, ProviderStrategy, ResponseFormat,
    StructuredOutputValidationError, ToolStrategy,
};
use langchain::chat_models::ParrotChatModel;
use langchain::messages::ToolCall;
use langchain::tools::{BaseTool, Tool, ToolDefinition, ToolRuntime};
use langchain_core::language_models::StructuredOutputSchema;
use langchain_core::messages::{AIMessage, HumanMessage, SystemMessage};
use serde_json::json;

struct NoopMiddleware;

impl AgentMiddleware for NoopMiddleware {}

#[test]
fn create_agent_is_reexported_through_factory_namespace() {
    let _agent = langchain::agents::factory::create_agent(ParrotChatModel::new("agent", 16));
    let _direct = create_agent(ParrotChatModel::new("agent", 16));
}

#[test]
fn structured_output_namespace_exposes_strategies_and_errors() {
    let schema = StructuredOutputSchema::new(
        "WeatherAnswer",
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        }),
    );

    let provider = ProviderStrategy::new(schema.clone()).with_strict(true);
    assert_eq!(provider.schema().name(), "WeatherAnswer");
    assert_eq!(provider.strict(), Some(true));

    let tool = ToolStrategy::new(schema.clone()).with_tool_message_content("done");
    assert_eq!(tool.schema().name(), "WeatherAnswer");
    assert_eq!(tool.tool_message_content(), Some("done"));

    let auto = AutoStrategy::new(schema.clone());
    assert_eq!(auto.schema().name(), "WeatherAnswer");

    let response_format = ResponseFormat::Provider(provider);
    assert_eq!(response_format.schema().name(), "WeatherAnswer");

    let multiple =
        MultipleStructuredOutputsError::new(vec!["tool_a".to_owned()], AIMessage::new(""));
    assert!(
        multiple
            .to_string()
            .contains("multiple structured responses")
    );

    let validation =
        StructuredOutputValidationError::new("tool_a", "bad payload", AIMessage::new(""));
    assert!(
        validation
            .to_string()
            .contains("Failed to parse structured output")
    );
}

#[test]
fn middleware_types_are_constructible_and_overrideable() {
    let model = Arc::new(ParrotChatModel::new("agent", 16));
    let request = ModelRequest::new(model.clone(), vec![HumanMessage::new("hello").into()])
        .with_system_message(SystemMessage::new("be concise"))
        .with_response_format(ResponseFormat::Auto(AutoStrategy::new(
            StructuredOutputSchema::new("Answer", json!({"type": "object"})),
        )))
        .with_state(AgentState::new(vec![HumanMessage::new("hello").into()]));

    assert_eq!(request.messages().len(), 1);
    assert_eq!(
        request
            .system_message()
            .expect("system message should exist")
            .content(),
        "be concise"
    );

    let overridden = request.override_with().with_jump_to(JumpTo::Tools);
    assert_eq!(overridden.jump_to(), Some(JumpTo::Tools));

    let response = ModelResponse::new(vec![AIMessage::new("done").into()])
        .with_structured_response(json!({"answer": "done"}));
    assert_eq!(response.result()[0].content(), "done");

    let extended = ExtendedModelResponse::new(response.clone()).with_jump_to(JumpTo::End);
    assert_eq!(extended.model_response().result()[0].content(), "done");
    assert_eq!(extended.jump_to(), Some(JumpTo::End));

    let middleware = NoopMiddleware;
    let mut state = request.state().clone();
    let maybe_jump = middleware
        .before_agent(&mut state)
        .expect("noop middleware should not fail");
    assert_eq!(maybe_jump, None);
}

#[test]
fn middleware_root_namespace_reexports_core_types_and_helpers() {
    use langchain::agents::middleware::{
        AgentMiddleware, AgentState as RootAgentState,
        ExtendedModelResponse as RootExtendedModelResponse, ModelCallResult,
        ModelRequest as RootModelRequest, ModelResponse as RootModelResponse,
        ToolCallRequest as RootToolCallRequest, after_agent, after_model, before_agent,
        before_model, dynamic_prompt, hook_config,
    };

    let model = Arc::new(ParrotChatModel::new("agent", 16));
    let request = RootModelRequest::new(model, vec![HumanMessage::new("hello").into()])
        .with_system_message(dynamic_prompt("be concise"))
        .with_state(RootAgentState::new(vec![HumanMessage::new("hello").into()]));

    let response: ModelCallResult = RootModelResponse::new(vec![AIMessage::new("done").into()]);
    let extended = RootExtendedModelResponse::new(response.clone());
    let hook = hook_config([]);

    assert_eq!(request.messages().len(), 1);
    assert_eq!(
        request
            .system_message()
            .expect("system message should exist")
            .content(),
        "be concise"
    );
    assert_eq!(extended.model_response().result()[0].content(), "done");
    assert!(hook.can_jump_to().is_empty());

    assert_eq!(
        before_agent(NoopMiddleware).name(),
        std::any::type_name::<NoopMiddleware>()
    );
    assert_eq!(
        before_model(NoopMiddleware).name(),
        std::any::type_name::<NoopMiddleware>()
    );
    assert_eq!(
        after_model(NoopMiddleware).name(),
        std::any::type_name::<NoopMiddleware>()
    );
    assert_eq!(
        after_agent(NoopMiddleware).name(),
        std::any::type_name::<NoopMiddleware>()
    );

    let tool: Arc<dyn BaseTool> = Arc::new(Tool::new(
        ToolDefinition::new("lookup", "Look up a record"),
        |_input| Box::pin(async move { Ok("done".to_owned()) }),
    ));
    let tool_request = RootToolCallRequest::new(
        ToolCall::new("lookup", json!({"input": "rust"})).with_id("call_lookup_root_1"),
        json!({"messages": ["hello"]}),
        ToolRuntime::new(json!({"messages": []}), json!({"writes": []}))
            .with_tool_call_id("call_lookup_root_1"),
    )
    .with_tool(tool.clone());

    assert_eq!(tool_request.tool_call().name(), "lookup");
    assert!(Arc::ptr_eq(
        tool_request.tool().expect("tool should be attached"),
        &tool,
    ));
}

#[test]
fn middleware_types_reexport_tool_call_request_surface() {
    let tool: Arc<dyn BaseTool> = Arc::new(Tool::new(
        ToolDefinition::new("lookup", "Look up a record"),
        |_input| Box::pin(async move { Ok("done".to_owned()) }),
    ));
    let request = ToolCallRequest::new(
        ToolCall::new("lookup", json!({"input": "rust"})).with_id("call_lookup_1"),
        json!({"messages": ["hello"]}),
        ToolRuntime::new(json!({"messages": []}), json!({"writes": []}))
            .with_tool_call_id("call_lookup_1"),
    )
    .with_tool(tool.clone());

    let updated = request
        .override_with()
        .with_state(json!({"messages": ["updated"]}));

    assert_eq!(request.tool_call().name(), "lookup");
    assert_eq!(updated.state()["messages"][0], "updated");
    assert!(Arc::ptr_eq(
        request.tool().expect("tool should be attached"),
        &tool,
    ));

    let wrapper: ToolCallWrapper = Arc::new(|request, handler| handler(request));
    let _ = wrapper;
}
