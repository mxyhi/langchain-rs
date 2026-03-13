use langchain_anthropic::middleware::AnthropicPromptCachingMiddleware;
use langchain_anthropic::{ChatAnthropic, convert_to_anthropic_tool};
use langchain_core::language_models::BaseChatModel;
use langchain_core::language_models::{ToolBindingOptions, ToolChoice};
use langchain_core::messages::HumanMessage;
use langchain_core::runnables::{Runnable, RunnableConfig};
use langchain_core::tools::tool;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn invokes_anthropic_messages_and_maps_text_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(header("x-api-key", "test-key"))
        .and(body_json(json!({
            "model": "claude-3-7-sonnet-latest",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "ping" }
                    ]
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_123",
            "model": "claude-3-7-sonnet-latest",
            "content": [
                { "type": "text", "text": "pong" }
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 4,
                "output_tokens": 1
            }
        })))
        .mount(&server)
        .await;

    let model = ChatAnthropic::new("claude-3-7-sonnet-latest", server.uri(), Some("test-key"));
    let message = model
        .invoke(vec![HumanMessage::new("ping").into()], Default::default())
        .await
        .expect("anthropic invocation should succeed");

    assert_eq!(message.content(), "pong");
    assert_eq!(
        message
            .usage_metadata()
            .expect("usage metadata should exist")
            .total_tokens,
        5
    );
}

#[tokio::test]
async fn bind_tools_serializes_anthropic_tool_schema_and_parses_tool_use() {
    let server = MockServer::start().await;

    let weather_tool = tool("get_weather", "Get the current weather").with_parameters(json!({
        "type": "object",
        "properties": {
            "city": { "type": "string" }
        },
        "required": ["city"]
    }));

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": "claude-3-7-sonnet-latest",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "weather?" }
                    ]
                }
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" }
                        },
                        "required": ["city"]
                    }
                }
            ],
            "tool_choice": { "type": "tool", "name": "get_weather" }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_tool_123",
            "model": "claude-3-7-sonnet-latest",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": { "city": "Boston" }
                }
            ]
        })))
        .mount(&server)
        .await;

    let bound_model = ChatAnthropic::new("claude-3-7-sonnet-latest", server.uri(), None::<&str>)
        .bind_tools(
            vec![weather_tool],
            ToolBindingOptions {
                tool_choice: Some(ToolChoice::Named("get_weather".to_owned())),
                ..ToolBindingOptions::default()
            },
        )
        .expect("binding tools should succeed");

    let message = bound_model
        .invoke(
            vec![HumanMessage::new("weather?").into()],
            Default::default(),
        )
        .await
        .expect("tool call should parse");

    assert_eq!(message.tool_calls().len(), 1);
    assert_eq!(message.tool_calls()[0].name(), "get_weather");
    assert_eq!(message.tool_calls()[0].args(), &json!({ "city": "Boston" }));

    let converted = convert_to_anthropic_tool(&tool("lookup", "Look up data"));
    assert_eq!(
        serde_json::to_value(converted).expect("tool should serialize"),
        json!({
            "name": "lookup",
            "description": "Look up data",
            "input_schema": {
                "type": "object",
                "properties": {}
            }
        })
    );
}

#[tokio::test]
async fn prompt_caching_metadata_serializes_into_messages_request() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": "claude-3-7-sonnet-latest",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "cache this prompt" }
                    ]
                }
            ],
            "cache_control": {
                "type": "ephemeral",
                "ttl": "1h"
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_cache_123",
            "model": "claude-3-7-sonnet-latest",
            "content": [
                { "type": "text", "text": "cached" }
            ]
        })))
        .mount(&server)
        .await;

    let model = ChatAnthropic::new("claude-3-7-sonnet-latest", server.uri(), None::<&str>);
    let config = AnthropicPromptCachingMiddleware::new()
        .with_ttl("1h")
        .with_min_messages_to_cache(1)
        .configured_config(1, RunnableConfig::default())
        .expect("cache middleware should produce Anthropic cache metadata");

    let message = model
        .invoke(vec![HumanMessage::new("cache this prompt").into()], config)
        .await
        .expect("anthropic invocation with cache metadata should succeed");

    assert_eq!(message.content(), "cached");
}
