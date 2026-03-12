use langchain_core::messages::HumanMessage;
use langchain_core::runnables::Runnable;
use langchain_core::tools::tool;
use langchain_openai::ChatOpenAI;
use serde_json::json;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn invokes_chat_completions_and_maps_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_test",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "pong"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 1,
                "total_tokens": 5
            }
        })))
        .mount(&server)
        .await;

    let model = ChatOpenAI::new("gpt-4o-mini", server.uri(), Some("test-key"));
    let message = model
        .invoke(vec![HumanMessage::new("ping").into()], Default::default())
        .await
        .expect("openai invocation should succeed");

    assert_eq!(message.content(), "pong");
    assert_eq!(
        message
            .response_metadata()
            .get("model")
            .and_then(|value| value.as_str()),
        Some("gpt-4o-mini")
    );
    assert_eq!(
        message
            .usage_metadata()
            .expect("usage metadata")
            .total_tokens,
        5
    );
}

#[tokio::test]
async fn bind_tools_sends_openai_tool_schema_and_parses_tool_calls() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_json(json!({
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": "what is the weather?"
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": { "type": "string" }
                            },
                            "required": ["city"]
                        }
                    }
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {
                    "name": "get_weather"
                }
            },
            "parallel_tool_calls": false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_tool",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_weather_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"city\":\"Boston\"}"
                                }
                            }
                        ]
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let model = ChatOpenAI::new("gpt-4o-mini", server.uri(), Some("test-key"))
        .bind_tools(vec![
            tool("get_weather", "Get the current weather").with_parameters(json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            })),
        ])
        .with_tool_choice("get_weather")
        .with_parallel_tool_calls(false);

    let message = model
        .invoke(
            vec![HumanMessage::new("what is the weather?").into()],
            Default::default(),
        )
        .await
        .expect("tool-bound invocation should succeed");

    assert_eq!(message.content(), "");
    assert_eq!(message.tool_calls().len(), 1);
    assert_eq!(message.tool_calls()[0].id(), Some("call_weather_1"));
    assert_eq!(message.tool_calls()[0].name(), "get_weather");
    assert_eq!(message.tool_calls()[0].args(), &json!({ "city": "Boston" }));
}
