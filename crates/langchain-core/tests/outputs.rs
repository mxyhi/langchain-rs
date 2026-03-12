use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::language_models::BaseLLM;
use langchain_core::messages::AIMessage;
use langchain_core::outputs::{
    ChatGeneration, Generation, GenerationCandidate, GenerationChunk, LLMGeneration, LLMResult,
};
use langchain_core::runnables::{Runnable, RunnableConfig};
use serde_json::json;

struct EchoLlm;

impl BaseLLM for EchoLlm {
    fn model_name(&self) -> &str {
        "echo-llm"
    }

    fn generate<'a>(
        &'a self,
        prompts: Vec<String>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<LLMResult, langchain_core::LangChainError>> {
        Box::pin(async move {
            Ok(LLMResult::new(
                prompts
                    .into_iter()
                    .map(|prompt| vec![GenerationCandidate::from(Generation::new(prompt))])
                    .collect(),
            ))
        })
    }
}

#[test]
fn generation_chunks_concatenate_text_and_metadata() {
    let left = GenerationChunk::with_info(
        "hel",
        BTreeMap::from([("finish_reason".to_owned(), json!("length"))]),
    );
    let right = GenerationChunk::with_info(
        "lo",
        BTreeMap::from([("provider".to_owned(), json!("openai"))]),
    );

    let merged = left + right;

    assert_eq!(merged.text(), "hello");
    assert_eq!(
        merged.generation_info(),
        Some(&BTreeMap::from([
            ("finish_reason".to_owned(), json!("length")),
            ("provider".to_owned(), json!("openai")),
        ]))
    );
}

#[test]
fn chat_generation_tracks_message_text() {
    let generation = ChatGeneration::new(AIMessage::new("pong"));

    assert_eq!(generation.text(), "pong");
    assert_eq!(generation.message().content(), "pong");
}

#[test]
fn llm_result_flatten_clears_token_usage_after_first_prompt() {
    let result = LLMResult::new(vec![
        vec![GenerationCandidate::from(Generation::new("alpha"))],
        vec![GenerationCandidate::from(Generation::new("beta"))],
    ])
    .with_output(BTreeMap::from([
        ("token_usage".to_owned(), json!({ "prompt_tokens": 6 })),
        ("model".to_owned(), json!("gpt-4o-mini")),
    ]));

    let flattened = result.flatten();

    assert_eq!(flattened.len(), 2);
    assert_eq!(
        flattened[0].llm_output(),
        Some(&BTreeMap::from([
            ("token_usage".to_owned(), json!({ "prompt_tokens": 6 })),
            ("model".to_owned(), json!("gpt-4o-mini")),
        ]))
    );
    assert_eq!(
        flattened[1].llm_output(),
        Some(&BTreeMap::from([
            ("token_usage".to_owned(), json!({})),
            ("model".to_owned(), json!("gpt-4o-mini")),
        ]))
    );
}

#[tokio::test]
async fn base_llm_is_runnable_and_returns_first_generation_text() {
    let model = EchoLlm;
    let output = model
        .invoke("langchain".to_owned(), RunnableConfig::default())
        .await
        .expect("llm invocation should succeed");

    assert_eq!(output, "langchain");
}

#[test]
fn llm_generation_exposes_text_for_chat_and_text_variants() {
    let text_generation = LLMGeneration::from(Generation::new("plain"));
    let chat_generation = LLMGeneration::from(ChatGeneration::new(AIMessage::new("chat")));

    assert_eq!(text_generation.text(), "plain");
    assert_eq!(chat_generation.text(), "chat");
    assert!(matches!(chat_generation, LLMGeneration::Chat(_)));
}

#[test]
fn llm_result_primary_generation_returns_first_candidate() {
    let result = LLMResult::new(vec![vec![GenerationCandidate::from(Generation::new(
        "first",
    ))]]);

    let generation = result
        .primary_generation()
        .expect("primary generation should exist");

    assert_eq!(generation.text(), "first");
}
