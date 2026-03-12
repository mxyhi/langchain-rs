use std::sync::Arc;

use langchain::agents::create_agent;
use langchain::agents::middleware::types::{
    AgentMiddleware, AgentState, ExtendedModelResponse, JumpTo, ModelRequest, ModelResponse,
};
use langchain::agents::structured_output::{
    AutoStrategy, MultipleStructuredOutputsError, ProviderStrategy, ResponseFormat,
    StructuredOutputValidationError, ToolStrategy,
};
use langchain::chat_models::ParrotChatModel;
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
    let maybe_jump = middleware.before_agent(request.state());
    assert_eq!(maybe_jump, None);
}
