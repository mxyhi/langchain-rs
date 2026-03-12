use std::collections::HashMap;

use langchain_classic::{ConversationChain, LLMChain};
use langchain_core::language_models::{ParrotChatModel, ParrotLLM};
use langchain_core::messages::{BaseMessage, HumanMessage};
use langchain_core::prompts::{PromptArgument, PromptTemplate};
use langchain_core::runnables::Runnable;

#[tokio::test]
async fn llm_chain_formats_prompt_and_invokes_model() {
    let chain = LLMChain::new(
        ParrotLLM::new("parrot-llm", 12),
        PromptTemplate::new("Topic: {topic}"),
    );
    let arguments = HashMap::from([(
        "topic".to_owned(),
        PromptArgument::String("rust".to_owned()),
    )]);

    let output = chain
        .invoke(arguments, Default::default())
        .await
        .expect("llm chain should succeed");

    assert_eq!(output, "Topic: rust");
}

#[tokio::test]
async fn conversation_chain_uses_history_and_latest_input() {
    let chain = ConversationChain::new(ParrotChatModel::new("parrot-chat", 16));
    let history = vec![BaseMessage::from(HumanMessage::new("hello there"))];

    let output = chain
        .predict("how are you?", history)
        .await
        .expect("conversation chain should succeed");

    assert_eq!(output, "how are you?");
}
