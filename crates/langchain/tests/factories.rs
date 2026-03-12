use std::collections::BTreeMap;

use langchain::embeddings::Embeddings;
use langchain::language_models::{
    BaseChatModel, StructuredOutput, StructuredOutputOptions, StructuredOutputSchema,
    ToolBindingOptions, ToolChoice,
};
use langchain::messages::HumanMessage;
use langchain::runnables::Runnable;
use langchain::tools::tool;
use langchain::{ModelInitOptions, init_chat_model, init_configurable_chat_model, init_embeddings};
use serde_json::json;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn init_chat_model_supports_provider_prefixed_model_name() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_json(json!({
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": "ping"
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_factory",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "pong"
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let model = init_chat_model(
        "openai:gpt-4o-mini",
        ModelInitOptions::default().with_base_url(server.uri()),
    )
    .expect("factory should create chat model");
    let message = model
        .invoke(vec![HumanMessage::new("ping").into()], Default::default())
        .await
        .expect("chat model should invoke");

    assert_eq!(message.content(), "pong");
}

#[tokio::test]
async fn configurable_chat_model_replays_bound_tools_at_runtime() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_json(json!({
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": "check the weather"
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
            "id": "chatcmpl_configurable_tool",
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

    let configurable = init_configurable_chat_model(
        None,
        ModelInitOptions::default().with_base_url(server.uri()),
    );
    assert_eq!(configurable.queued_operation_count(), 0);

    let bound = configurable.clone().bind_tools(
        vec![
            tool("get_weather", "Get the current weather").with_parameters(json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            })),
        ],
        ToolBindingOptions {
            tool_choice: Some(ToolChoice::Named("get_weather".to_owned())),
            parallel_tool_calls: Some(false),
            ..ToolBindingOptions::default()
        },
    );
    assert_eq!(configurable.queued_operation_count(), 0);
    assert_eq!(bound.queued_operation_count(), 1);

    let response = bound
        .invoke(
            vec![HumanMessage::new("check the weather").into()],
            langchain::runnables::RunnableConfig {
                configurable: BTreeMap::from([("model".to_owned(), json!("gpt-4o-mini"))]),
                ..Default::default()
            },
        )
        .await
        .expect("configurable model should resolve and invoke");

    assert_eq!(response.tool_calls().len(), 1);
    assert_eq!(response.tool_calls()[0].name(), "get_weather");
}

#[tokio::test]
async fn configurable_chat_model_supports_structured_output() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_json(json!({
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": "answer the question"
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "AnswerPayload",
                        "description": "Structured answer payload",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": { "type": "string" }
                            },
                            "required": ["answer"]
                        }
                    }
                }
            ],
            "tool_choice": "required",
            "parallel_tool_calls": false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_configurable_structured",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_answer_1",
                                "type": "function",
                                "function": {
                                    "name": "AnswerPayload",
                                    "arguments": "{\"answer\":\"Boston\"}"
                                }
                            }
                        ]
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let configurable = init_configurable_chat_model(
        None,
        ModelInitOptions::default().with_base_url(server.uri()),
    );
    let runnable = configurable.with_structured_output(
        StructuredOutputSchema::new(
            "AnswerPayload",
            json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                },
                "required": ["answer"]
            }),
        )
        .with_description("Structured answer payload"),
        StructuredOutputOptions::default(),
    );

    let output = runnable
        .invoke(
            vec![HumanMessage::new("answer the question").into()],
            langchain::runnables::RunnableConfig {
                configurable: BTreeMap::from([("model".to_owned(), json!("gpt-4o-mini"))]),
                ..Default::default()
            },
        )
        .await
        .expect("configurable structured output should succeed");

    match output {
        StructuredOutput::Parsed(value) => {
            assert_eq!(value, json!({ "answer": "Boston" }));
        }
        StructuredOutput::Raw { .. } => panic!("expected parsed structured output"),
    }
}

#[tokio::test]
async fn init_chat_model_infers_openai_provider_from_model_prefix() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_infer",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "pong"
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let model = init_chat_model(
        "gpt-4o-mini",
        ModelInitOptions::default().with_base_url(server.uri()),
    )
    .expect("provider inference should succeed");
    let message = model
        .invoke(vec![HumanMessage::new("ping").into()], Default::default())
        .await
        .expect("chat model should invoke");

    assert_eq!(message.content(), "pong");
}

#[tokio::test]
async fn init_embeddings_supports_explicit_provider() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "embedding": [0.1, 0.2, 0.3] }
            ]
        })))
        .mount(&server)
        .await;

    let embeddings = init_embeddings(
        "text-embedding-3-small",
        ModelInitOptions::default()
            .with_provider("openai")
            .with_base_url(server.uri()),
    )
    .expect("embedding factory should create provider instance");
    let result = embeddings
        .embed_query("hello")
        .await
        .expect("embedding request should succeed");

    assert_eq!(result, vec![0.1, 0.2, 0.3]);
}

#[test]
fn init_embeddings_requires_provider_when_model_name_cannot_be_inferred() {
    let error = match init_embeddings("text-embedding-3-small", ModelInitOptions::default()) {
        Ok(_) => panic!("missing provider should fail"),
        Err(error) => error,
    };

    assert_eq!(
        error.to_string(),
        "unsupported operation: must specify provider or use `provider:model` format for embeddings"
    );
}

#[test]
fn init_chat_model_rejects_provider_boundaries_without_runtime_transport() {
    for provider in ["huggingface", "perplexity"] {
        let error = match init_chat_model(
            "test-model",
            ModelInitOptions::default().with_provider(provider),
        ) {
            Ok(_) => panic!("provider {provider} should be rejected until transport exists"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains(&format!("Unsupported provider='{provider}'")),
            "unexpected error for {provider}: {error}"
        );
    }
}

#[test]
fn init_embeddings_rejects_provider_boundaries_without_runtime_transport() {
    for provider in ["huggingface", "nomic"] {
        let error = match init_embeddings(
            "test-embedding-model",
            ModelInitOptions::default().with_provider(provider),
        ) {
            Ok(_) => panic!("provider {provider} should be rejected until transport exists"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains(&format!("Provider '{provider}' is not supported")),
            "unexpected error for {provider}: {error}"
        );
    }
}

#[tokio::test]
async fn init_chat_model_trait_object_supports_structured_output() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_factory_structured",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_answer_1",
                                "type": "function",
                                "function": {
                                    "name": "AnswerPayload",
                                    "arguments": "{\"answer\":\"Boston\"}"
                                }
                            }
                        ]
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let model = init_chat_model(
        "openai:gpt-4o-mini",
        ModelInitOptions::default().with_base_url(server.uri()),
    )
    .expect("factory should create chat model");
    let runnable = model
        .with_structured_output(
            StructuredOutputSchema::new(
                "AnswerPayload",
                json!({
                    "type": "object",
                    "properties": {
                        "answer": { "type": "string" }
                    },
                    "required": ["answer"]
                }),
            )
            .with_description("Structured answer payload"),
            StructuredOutputOptions::default(),
        )
        .expect("structured output should be supported");

    let output = runnable
        .invoke_boxed(
            vec![HumanMessage::new("answer the question").into()],
            Default::default(),
        )
        .await
        .expect("structured output invocation should succeed");

    match output {
        StructuredOutput::Parsed(value) => {
            assert_eq!(value, json!({ "answer": "Boston" }));
        }
        StructuredOutput::Raw { .. } => panic!("expected parsed structured output"),
    }

    let requests = server
        .received_requests()
        .await
        .expect("wiremock should expose received requests");
    let request_body: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("request body should be json");

    assert_eq!(requests[0].url.path(), "/chat/completions");
    assert_eq!(
        request_body,
        json!({
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": "answer the question"
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "AnswerPayload",
                        "description": "Structured answer payload",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": { "type": "string" }
                            },
                            "required": ["answer"]
                        }
                    }
                }
            ],
            "tool_choice": "required",
            "parallel_tool_calls": false
        })
    );
}

#[tokio::test]
async fn init_chat_model_infers_anthropic_provider_from_model_prefix() {
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
                        { "type": "text", "text": "ping" }
                    ]
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_factory_anthropic",
            "model": "claude-3-7-sonnet-latest",
            "content": [
                { "type": "text", "text": "pong" }
            ]
        })))
        .mount(&server)
        .await;

    let model = init_chat_model(
        "claude-3-7-sonnet-latest",
        ModelInitOptions::default().with_base_url(server.uri()),
    )
    .expect("anthropic inference should succeed");
    let response = model
        .invoke(vec![HumanMessage::new("ping").into()], Default::default())
        .await
        .expect("anthropic chat model should invoke");

    assert_eq!(response.content(), "pong");
}

#[tokio::test]
async fn configurable_chat_model_runtime_base_url_override_beats_provider_default() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_json(json!({
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": "ping"
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_runtime_override",
            "model": "deepseek-chat",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "pong"
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let configurable = init_configurable_chat_model(None, ModelInitOptions::default());
    let response = configurable
        .invoke(
            vec![HumanMessage::new("ping").into()],
            langchain::runnables::RunnableConfig {
                configurable: BTreeMap::from([
                    ("model".to_owned(), json!("deepseek-chat")),
                    ("provider".to_owned(), json!("deepseek")),
                    ("base_url".to_owned(), json!(server.uri())),
                ]),
                ..Default::default()
            },
        )
        .await
        .expect("runtime base_url override should route request to mock server");

    assert_eq!(response.content(), "pong");
}

#[tokio::test]
async fn init_embeddings_supports_registered_provider_default_resolution() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(body_json(json!({
            "model": "mistral-embed",
            "input": ["hello"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "embedding": [0.8, 0.9] }
            ]
        })))
        .mount(&server)
        .await;

    let embeddings = init_embeddings(
        "mistralai:mistral-embed",
        ModelInitOptions::default().with_base_url(server.uri()),
    )
    .expect("registered embeddings provider should resolve");
    let response = embeddings
        .embed_query("hello")
        .await
        .expect("embedding query should succeed");

    assert_eq!(response, vec![0.8, 0.9]);
}
