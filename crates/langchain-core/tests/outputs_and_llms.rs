use std::collections::BTreeMap;

use langchain_core::language_models::{BaseLLM, ParrotLLM};
use langchain_core::outputs::{
    ChatGeneration, ChatGenerationChunk, Generation, GenerationCandidate, GenerationChunk,
    LLMResult,
};
use langchain_core::runnables::{Runnable, RunnableConfig};
use langchain_core::{messages::AIMessage, messages::UsageMetadata};
use serde_json::json;

#[test]
fn generation_chunk_concatenates_text_and_metadata() {
    let left = GenerationChunk::with_info(
        "hel",
        BTreeMap::from([("finish_reason".to_owned(), json!(null))]),
    );
    let right = GenerationChunk::with_info(
        "lo",
        BTreeMap::from([("logprobs".to_owned(), json!({"token": "o"}))]),
    );

    let chunk = left + right;

    assert_eq!(chunk.text(), "hello");
    assert_eq!(
        chunk
            .generation_info()
            .and_then(|info| info.get("logprobs")),
        Some(&json!({"token": "o"}))
    );
}

#[test]
fn chat_generation_tracks_message_text() {
    let message = AIMessage::new("pong");
    let generation = ChatGeneration::new(message);

    assert_eq!(generation.text(), "pong");
}

#[test]
fn chat_generation_chunk_concatenates_messages() {
    let left = ChatGenerationChunk::new(AIMessage::with_metadata(
        "hel",
        BTreeMap::from([("id".to_owned(), json!("left"))]),
        Some(UsageMetadata {
            input_tokens: 1,
            output_tokens: 2,
            total_tokens: 3,
        }),
    ));
    let right = ChatGenerationChunk::with_info(
        AIMessage::with_metadata(
            "lo",
            BTreeMap::from([("model".to_owned(), json!("gpt"))]),
            Some(UsageMetadata {
                input_tokens: 2,
                output_tokens: 3,
                total_tokens: 5,
            }),
        ),
        BTreeMap::from([("finish_reason".to_owned(), json!("stop"))]),
    );

    let chunk = left + right;

    assert_eq!(chunk.text(), "hello");
    assert_eq!(
        chunk
            .message()
            .response_metadata()
            .get("model")
            .and_then(|value| value.as_str()),
        Some("gpt")
    );
    assert_eq!(
        chunk
            .message()
            .usage_metadata()
            .expect("usage metadata should be merged")
            .total_tokens,
        8
    );
}

#[test]
fn llm_result_flatten_keeps_token_usage_only_once() {
    let result = LLMResult::new(vec![
        vec![GenerationCandidate::Text(Generation::new("alpha"))],
        vec![GenerationCandidate::Text(Generation::new("beta"))],
    ])
    .with_output(BTreeMap::from([
        ("token_usage".to_owned(), json!({"prompt_tokens": 3})),
        ("model_name".to_owned(), json!("gpt-4o-mini")),
    ]));

    let flattened = result.flatten();

    assert_eq!(flattened.len(), 2);
    assert_eq!(
        flattened[0]
            .llm_output()
            .and_then(|output| output.get("token_usage")),
        Some(&json!({"prompt_tokens": 3}))
    );
    assert_eq!(
        flattened[1]
            .llm_output()
            .and_then(|output| output.get("token_usage")),
        Some(&json!({}))
    );
}

#[tokio::test]
async fn parrot_llm_is_runnable_over_plain_text() {
    let model = ParrotLLM::new("parrot-llm", 5);

    let output = model
        .invoke("langchain".to_owned(), RunnableConfig::default())
        .await
        .expect("llm invocation should succeed");
    let result = model
        .generate(vec!["langchain".to_owned()], RunnableConfig::default())
        .await
        .expect("llm generation should succeed");

    assert_eq!(output, "langc");
    assert_eq!(result.generations().len(), 1);
    assert_eq!(result.generations()[0][0].text(), "langc");
}
