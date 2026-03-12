use langchain_core::LangChainError;
use langchain_core::language_models::BaseLLM;
use langchain_core::outputs::LLMResult;
use langchain_core::runnables::{Runnable, RunnableConfig};

pub async fn assert_llm_invoke_response<M>(model: &M, prompt: &str, expected: &str)
where
    M: BaseLLM,
{
    let response = model
        .invoke(prompt.to_owned(), RunnableConfig::default())
        .await
        .expect("llm invoke should succeed");

    assert_eq!(response, expected);
}

pub async fn assert_llm_generate_texts<M>(
    model: &M,
    prompts: Vec<String>,
    expected: &[&str],
) -> LLMResult
where
    M: BaseLLM,
{
    let result = model
        .generate(prompts, RunnableConfig::default())
        .await
        .expect("llm generate should succeed");

    let texts = result
        .generations()
        .iter()
        .map(|generations| {
            generations
                .first()
                .expect("each prompt should have at least one generation")
                .text()
        })
        .collect::<Vec<_>>();

    assert_eq!(texts, expected);
    result
}

pub fn assert_llm_token_usage(
    llm_output: Option<&langchain_core::messages::ResponseMetadata>,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> Result<(), LangChainError> {
    let llm_output =
        llm_output.ok_or_else(|| LangChainError::unsupported("llm output metadata missing"))?;
    let token_usage = llm_output
        .get("token_usage")
        .ok_or_else(|| LangChainError::unsupported("llm token_usage missing"))?;

    assert_eq!(
        token_usage
            .get("prompt_tokens")
            .and_then(|value| value.as_u64()),
        Some(prompt_tokens as u64)
    );
    assert_eq!(
        token_usage
            .get("completion_tokens")
            .and_then(|value| value.as_u64()),
        Some(completion_tokens as u64)
    );
    assert_eq!(
        token_usage
            .get("total_tokens")
            .and_then(|value| value.as_u64()),
        Some((prompt_tokens + completion_tokens) as u64)
    );

    Ok(())
}
