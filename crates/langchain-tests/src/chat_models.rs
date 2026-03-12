use langchain_core::LangChainError;
use langchain_core::language_models::BaseChatModel;
use langchain_core::messages::BaseMessage;
use langchain_core::runnables::{Runnable, RunnableConfig};

pub async fn assert_chat_model_response<M>(model: &M, prompt: Vec<BaseMessage>, expected: &str)
where
    M: BaseChatModel,
{
    let message = model
        .generate(prompt, RunnableConfig::default())
        .await
        .expect("chat model should generate a response");

    assert_eq!(message.content(), expected);
}

pub async fn assert_chat_model_batch<M>(
    model: &M,
    prompts: Vec<Vec<BaseMessage>>,
    expected: &[&str],
) where
    M: BaseChatModel,
{
    let responses = model
        .batch(prompts, RunnableConfig::default())
        .await
        .expect("chat model batch should succeed");

    let contents = responses
        .iter()
        .map(|message| message.content())
        .collect::<Vec<_>>();

    assert_eq!(contents, expected);
}

pub fn assert_usage_tokens(
    usage: Option<&langchain_core::messages::UsageMetadata>,
    input_tokens: usize,
    output_tokens: usize,
) -> Result<(), LangChainError> {
    let usage = usage.ok_or_else(|| LangChainError::unsupported("usage metadata missing"))?;

    assert_eq!(usage.input_tokens, input_tokens);
    assert_eq!(usage.output_tokens, output_tokens);
    assert_eq!(usage.total_tokens, input_tokens + output_tokens);

    Ok(())
}
