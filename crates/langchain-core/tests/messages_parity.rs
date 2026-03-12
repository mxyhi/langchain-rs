use serde_json::json;

use langchain_core::messages::{
    AIMessage, AIMessageChunk, Annotation, BaseMessage, BaseMessageChunk, ChatMessage, Citation,
    ContentBlock, FunctionMessage, HumanMessage, MessageLikeRepresentation, RemoveMessage,
    ServerToolCall, ServerToolCallChunk, ServerToolResult, SystemMessage, TextContentBlock,
    ToolCall, ToolCallChunk, ToolMessage, ToolMessageChunk, ToolMessageStatus, convert_to_messages,
    convert_to_openai_messages, filter_messages, merge_message_runs, message_chunk_to_message,
    messages_from_dict, messages_to_dict,
};

#[test]
fn message_chunk_to_message_converts_chunk_variants() {
    let ai_chunk =
        BaseMessageChunk::Ai(AIMessageChunk::new("partial").with_tool_call_chunks(vec![
                ToolCallChunk::new()
                    .with_id("call_ok")
                    .with_name("lookup")
                    .with_args("{\"query\":\"rust\"}"),
                ToolCallChunk::new()
                    .with_id("call_bad")
                    .with_args("{not-json"),
            ]));
    let tool_chunk =
        BaseMessageChunk::Tool(ToolMessageChunk::new("done", "call_ok").with_name("lookup"));

    let ai_message = message_chunk_to_message(&ai_chunk);
    let tool_message = message_chunk_to_message(&tool_chunk);

    match ai_message {
        BaseMessage::Ai(message) => {
            assert_eq!(message.content(), "partial");
            assert_eq!(message.tool_calls().len(), 1);
            assert_eq!(message.tool_calls()[0].name(), "lookup");
            assert_eq!(message.invalid_tool_calls().len(), 1);
            assert_eq!(message.invalid_tool_calls()[0].id(), Some("call_bad"));
        }
        other => panic!("expected ai message, got {other:?}"),
    }

    match tool_message {
        BaseMessage::Tool(message) => {
            assert_eq!(message.content(), "done");
            assert_eq!(message.tool_call_id(), "call_ok");
            assert_eq!(message.name(), Some("lookup"));
        }
        other => panic!("expected tool message, got {other:?}"),
    }
}

#[test]
fn conversion_helpers_accept_strings_roles_dicts_and_chat_messages() {
    let messages = convert_to_messages(vec![
        MessageLikeRepresentation::from("hello"),
        MessageLikeRepresentation::RoleAndContent {
            role: "assistant".to_owned(),
            content: "pong".to_owned(),
        },
        MessageLikeRepresentation::Chat(ChatMessage::new("system", "be concise")),
        MessageLikeRepresentation::Function(FunctionMessage::new(
            "weather",
            "{\"city\":\"Paris\"}",
        )),
        MessageLikeRepresentation::Dict(json!({
            "role": "tool",
            "content": "sunny",
            "tool_call_id": "call_weather",
            "name": "weather"
        })),
    ])
    .expect("message conversion should succeed");

    assert!(matches!(messages[0], BaseMessage::Human(_)));
    assert!(matches!(messages[1], BaseMessage::Ai(_)));
    assert!(matches!(messages[2], BaseMessage::System(_)));
    assert!(matches!(messages[3], BaseMessage::Tool(_)));
    assert!(matches!(messages[4], BaseMessage::Tool(_)));
}

#[test]
fn dict_roundtrip_and_openai_conversion_keep_roles_and_tool_calls() {
    let original = vec![
        BaseMessage::from(HumanMessage::new("hello")),
        BaseMessage::from(SystemMessage::new("be precise")),
        BaseMessage::from(AIMessage::new("pong").with_tool_calls(vec![
            ToolCall::new("lookup", json!({"query":"rust"})).with_id("call_1"),
        ])),
        BaseMessage::from(ToolMessage::with_parts(
            "42",
            "call_1",
            Some("lookup"),
            Some(json!({"value": 42})),
            ToolMessageStatus::Success,
        )),
    ];

    let dicts = messages_to_dict(&original);
    let roundtrip = messages_from_dict(&dicts).expect("roundtrip should succeed");
    let openai = convert_to_openai_messages(&roundtrip);

    assert_eq!(roundtrip, original);
    assert_eq!(openai[0]["role"], "user");
    assert_eq!(openai[1]["role"], "system");
    assert_eq!(openai[2]["role"], "assistant");
    assert_eq!(openai[2]["tool_calls"][0]["function"]["name"], "lookup");
    assert_eq!(openai[3]["role"], "tool");
    assert_eq!(openai[3]["tool_call_id"], "call_1");
}

#[test]
fn filter_and_merge_helpers_cover_common_message_flows() {
    let merged = merge_message_runs(&[
        BaseMessage::from(HumanMessage::new("alpha")),
        BaseMessage::from(HumanMessage::new("beta")),
        BaseMessage::from(AIMessage::new("one")),
        BaseMessage::from(AIMessage::new("two")),
        BaseMessage::from(ToolMessage::new("tool output", "call_1")),
    ]);

    assert_eq!(merged.len(), 3);
    assert_eq!(merged[0].content(), "alpha\nbeta");
    assert_eq!(merged[1].content(), "one\ntwo");

    let filtered = filter_messages(
        &merged,
        &[
            langchain_core::messages::MessageRole::Ai,
            langchain_core::messages::MessageRole::Tool,
        ],
    );

    assert_eq!(filtered.len(), 2);
    assert!(matches!(filtered[0], BaseMessage::Ai(_)));
    assert!(matches!(filtered[1], BaseMessage::Tool(_)));
}

#[test]
fn extra_boundaries_are_constructible_and_serde_friendly() {
    let annotation = Annotation::new("confidence", json!(0.9));
    let citation = Citation::new("https://example.com").with_title("Example");
    let content = ContentBlock::Text(
        TextContentBlock::new("hello")
            .with_annotations(vec![annotation.clone()])
            .with_citations(vec![citation.clone()]),
    );
    let server_call = ServerToolCall::new("server_1", "web_search", json!({"query":"rust"}));
    let server_chunk = ServerToolCallChunk::new()
        .with_id("server_1")
        .with_name("web_search")
        .with_args("{\"query\":\"rust\"}");
    let server_result =
        ServerToolResult::new("server_1", json!({"hits": 3})).with_name("web_search");
    let remove = RemoveMessage::new("message_1");

    let encoded = serde_json::to_value((
        &content,
        &server_call,
        &server_chunk,
        &server_result,
        &remove,
    ))
    .expect("extra message boundaries should serialize");

    assert_eq!(annotation.kind(), "confidence");
    assert_eq!(citation.url(), "https://example.com");
    assert_eq!(server_call.name(), "web_search");
    assert_eq!(server_chunk.id(), Some("server_1"));
    assert_eq!(server_result.name(), Some("web_search"));
    assert_eq!(remove.id(), "message_1");
    assert!(encoded.is_array());
}
