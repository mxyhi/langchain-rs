use langchain_core::language_models::BaseChatModel;
use langchain_core::messages::BaseMessage;
use langchain_core::runnables::RunnableConfig;

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
