use langchain_core::language_models::StructuredOutput;
use langchain_core::messages::HumanMessage;
use langchain_core::runnables::Runnable;
use langchain_core::tools::tool;
use langchain_openai::{ChatOpenAI, StructuredOutputMethod};
use serde_json::json;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn structured_output_function_calling_parses_first_tool_call() {
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
            "id": "chatcmpl_structured_function",
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

    let runnable = ChatOpenAI::new("gpt-4o-mini", server.uri(), Some("test-key"))
        .with_structured_output(
            tool("AnswerPayload", "Structured answer payload").with_parameters(json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                },
                "required": ["answer"]
            })),
            StructuredOutputMethod::FunctionCalling,
            false,
        );

    let result = runnable
        .invoke(
            vec![HumanMessage::new("answer the question").into()],
            Default::default(),
        )
        .await
        .expect("structured output invocation should succeed");

    match result {
        StructuredOutput::Parsed(value) => {
            assert_eq!(value, json!({ "answer": "Boston" }));
        }
        StructuredOutput::Raw { .. } => panic!("expected parsed-only structured output"),
    }
}

#[tokio::test]
async fn structured_output_json_schema_sends_response_format_and_can_include_raw() {
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
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "AnswerPayload",
                    "description": "Structured answer payload",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": { "type": "string" }
                        },
                        "required": ["answer"]
                    },
                    "strict": true
                }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_structured_schema",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "{\"answer\":\"Boston\"}"
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let runnable = ChatOpenAI::new("gpt-4o-mini", server.uri(), Some("test-key"))
        .with_structured_output(
            tool("AnswerPayload", "Structured answer payload")
                .with_parameters(json!({
                    "type": "object",
                    "properties": {
                        "answer": { "type": "string" }
                    },
                    "required": ["answer"]
                }))
                .with_strict(true),
            StructuredOutputMethod::JsonSchema,
            true,
        );

    let result = runnable
        .invoke(
            vec![HumanMessage::new("answer the question").into()],
            Default::default(),
        )
        .await
        .expect("json schema structured output should succeed");

    match result {
        StructuredOutput::Raw {
            raw,
            parsed,
            parsing_error,
        } => {
            assert_eq!(parsing_error, None);
            assert_eq!(parsed, Some(json!({ "answer": "Boston" })));
            assert_eq!(raw.content(), "{\"answer\":\"Boston\"}");
        }
        StructuredOutput::Parsed(_) => panic!("expected raw structured output envelope"),
    }
}
