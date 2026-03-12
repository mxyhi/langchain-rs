use serde_json::json;

use langchain_core::messages::{
    AIMessage, AIMessageChunk, Annotation, AudioContentBlock, BaseMessage, BaseMessageChunk,
    ChatMessage, ChatMessageChunk, Citation, ContentBlock, FileContentBlock, FunctionMessage,
    FunctionMessageChunk, HumanMessage, ImageContentBlock, MessageLikeRepresentation,
    NonStandardContentBlock, PlainTextContentBlock, ReasoningContentBlock, RemoveMessage,
    ServerToolCall, ServerToolCallChunk, ServerToolResult, SystemMessage, TextContentBlock,
    ToolCall, ToolCallChunk, ToolMessage, ToolMessageChunk, ToolMessageStatus, VideoContentBlock,
    convert_to_messages, convert_to_openai_messages, filter_messages, get_buffer_string,
    is_data_content_block, merge_content, merge_message_runs, message_chunk_to_message,
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

#[test]
fn chat_and_function_message_chunks_merge_with_identity_guards() {
    let merged_chat = ChatMessageChunk::new("critic", "hel")
        .with_id("chat_1")
        .try_merge(&ChatMessageChunk::new("critic", "lo"))
        .expect("chat chunks with the same role should merge");
    let merged_function = FunctionMessageChunk::new("weather", "{\"city\":")
        .try_merge(&FunctionMessageChunk::new("weather", "\"Paris\"}"))
        .expect("function chunks with the same name should merge");

    assert_eq!(merged_chat.role(), "critic");
    assert_eq!(merged_chat.content(), "hello");
    assert_eq!(merged_chat.id(), Some("chat_1"));
    assert_eq!(
        merged_chat.to_message(),
        ChatMessage::new("critic", "hello")
    );

    assert_eq!(merged_function.name(), "weather");
    assert_eq!(merged_function.content(), "{\"city\":\"Paris\"}");
    assert_eq!(
        merged_function.to_message(),
        FunctionMessage::new("weather", "{\"city\":\"Paris\"}")
    );

    let chat_error = ChatMessageChunk::new("critic", "a")
        .try_merge(&ChatMessageChunk::new("assistant", "b"))
        .expect_err("different roles should be rejected");
    let function_error = FunctionMessageChunk::new("weather", "a")
        .try_merge(&FunctionMessageChunk::new("calendar", "b"))
        .expect_err("different function names should be rejected");

    assert!(chat_error.to_string().contains("different roles"));
    assert!(function_error.to_string().contains("different names"));
}

#[test]
fn multimodal_content_blocks_and_helpers_cover_parity_surface() {
    let text = ContentBlock::Text(TextContentBlock::new("intro"));
    let reasoning = ContentBlock::Reasoning(ReasoningContentBlock::new("thinking"));
    let plain = ContentBlock::PlainText(PlainTextContentBlock::new("doc body"));
    let image = ContentBlock::Image(ImageContentBlock::new().with_url("https://example.com/a.png"));
    let audio = ContentBlock::Audio(AudioContentBlock::new().with_file_id("file_audio_1"));
    let file = ContentBlock::File(FileContentBlock::new().with_file_id("file_doc_1"));
    let video = ContentBlock::Video(VideoContentBlock::new().with_url("https://example.com/a.mp4"));
    let non_standard = ContentBlock::NonStandard(NonStandardContentBlock::new(json!({
        "provider": "custom",
        "payload": { "token": 7 }
    })));

    let merged_text = merge_content(String::from("hel"), String::from("lo"));
    let merged_blocks = merge_content(
        vec![text.clone(), reasoning.clone()],
        vec![
            plain.clone(),
            image.clone(),
            audio.clone(),
            file.clone(),
            video.clone(),
            non_standard.clone(),
        ],
    );
    let encoded = serde_json::to_value(&merged_blocks).expect("content blocks should serialize");

    assert_eq!(merged_text, "hello");
    assert_eq!(merged_blocks.len(), 8);
    assert!(!is_data_content_block(&text));
    assert!(!is_data_content_block(&reasoning));
    assert!(is_data_content_block(&plain));
    assert!(is_data_content_block(&image));
    assert!(is_data_content_block(&audio));
    assert!(is_data_content_block(&file));
    assert!(is_data_content_block(&video));
    assert!(!is_data_content_block(&non_standard));
    assert_eq!(encoded[0]["type"], "text");
    assert_eq!(encoded[2]["type"], "plain_text");
    assert_eq!(encoded[7]["type"], "non_standard");
}

#[test]
fn get_buffer_string_formats_base_chat_and_function_messages() {
    let buffer = get_buffer_string(vec![
        MessageLikeRepresentation::from(HumanMessage::new("hello")),
        MessageLikeRepresentation::from(AIMessage::new("pong")),
        MessageLikeRepresentation::from(ChatMessage::new("critic", "needs citation")),
        MessageLikeRepresentation::from(FunctionMessage::new("weather", "{\"city\":\"Paris\"}")),
    ])
    .expect("buffer helper should support parity message surface");

    assert_eq!(
        buffer,
        "Human: hello\nAI: pong\ncritic: needs citation\nFunction[weather]: {\"city\":\"Paris\"}"
    );
}
