use std::collections::BTreeMap;

use langchain_core::language_models::{BaseLLM, ParrotLLM};
use langchain_core::messages::AIMessage;
use langchain_core::outputs::{ChatGeneration, Generation, GenerationChunk, LLMResult};
use langchain_core::runnables::{Runnable, RunnableConfig};
use serde_json::json;

#[tokio::test]
async fn parrot_llm_invoke_returns_truncated_text() {
    let model = ParrotLLM::new("parrot-llm", 6);

    let output = model
        .invoke(String::from("langchain"), RunnableConfig::default())
        .await
        .expect("llm invoke should succeed");

    assert_eq!(output, "langch");
}

#[tokio::test]
async fn parrot_llm_generate_returns_llm_result_and_usage() {
    let model = ParrotLLM::new("parrot-llm", 5);

    let result = model
        .generate(
            vec![String::from("alpha"), String::from("beta")],
            RunnableConfig::default(),
        )
        .await
        .expect("llm generate should succeed");

    assert_eq!(result.generations()[0][0].text(), "alpha");
    assert_eq!(result.generations()[1][0].text(), "beta");
    assert_eq!(
        result.llm_output().and_then(|output| output.get("model")),
        Some(&json!("parrot-llm"))
    );
    assert_eq!(
        result
            .llm_output()
            .and_then(|output| output.get("token_usage"))
            .cloned(),
        Some(json!({
            "prompt_tokens": 9,
            "completion_tokens": 9,
            "total_tokens": 18
        }))
    );
}

#[test]
fn generation_chunk_add_preserves_text_and_merges_metadata() {
    let first = GenerationChunk::with_info(
        "hel",
        BTreeMap::from([
            ("index".to_owned(), json!(0)),
            ("finish_reason".to_owned(), json!(null)),
        ]),
    );
    let second = GenerationChunk::with_info(
        "lo",
        BTreeMap::from([("finish_reason".to_owned(), json!("stop"))]),
    );

    let merged = first + second;

    assert_eq!(merged.text(), "hello");
    assert_eq!(
        merged.generation_info().and_then(|info| info.get("index")),
        Some(&json!(0))
    );
    assert_eq!(
        merged
            .generation_info()
            .and_then(|info| info.get("finish_reason")),
        Some(&json!("stop"))
    );
}

#[test]
fn chat_generation_uses_ai_message_content_as_text() {
    let message = AIMessage::new("pong");
    let generation = ChatGeneration::new(message.clone());

    assert_eq!(generation.text(), "pong");
    assert_eq!(generation.message(), &message);
}

#[test]
fn llm_result_flatten_keeps_token_usage_only_on_first_item() {
    let result = LLMResult::new(vec![
        vec![Generation::new("one")],
        vec![Generation::new("two")],
    ])
    .with_output(BTreeMap::from([
        ("model".to_owned(), json!("openai")),
        (
            "token_usage".to_owned(),
            json!({
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "total_tokens": 4
            }),
        ),
    ]));

    let flattened = result.flatten();

    assert_eq!(flattened.len(), 2);
    assert_eq!(
        flattened[0]
            .llm_output()
            .and_then(|output| output.get("token_usage"))
            .cloned(),
        Some(json!({
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "total_tokens": 4
        }))
    );
    assert_eq!(
        flattened[1]
            .llm_output()
            .and_then(|output| output.get("token_usage"))
            .cloned(),
        Some(json!({}))
    );
}
