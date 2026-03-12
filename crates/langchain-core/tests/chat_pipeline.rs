use std::collections::HashMap;

use langchain_core::language_models::ParrotChatModel;
use langchain_core::messages::{BaseMessage, HumanMessage};
use langchain_core::output_parsers::StrOutputParser;
use langchain_core::prompts::{ChatPromptTemplate, PromptArgument, PromptMessageTemplate};
use langchain_core::runnables::{Runnable, RunnableConfig, RunnablePassthrough};

#[tokio::test]
async fn prompt_model_parser_pipeline_produces_text() {
    let prompt = ChatPromptTemplate::from_messages([
        PromptMessageTemplate::system("You are a {role}."),
        PromptMessageTemplate::human("{question}"),
    ]);
    let model = ParrotChatModel::new("parrot-1", 6);
    let parser = StrOutputParser::new();
    let chain = prompt.pipe(model).pipe(parser);

    let mut args = HashMap::new();
    args.insert(
        "role".to_owned(),
        PromptArgument::String("translator".to_owned()),
    );
    args.insert(
        "question".to_owned(),
        PromptArgument::String("langchain".to_owned()),
    );

    let output = chain
        .invoke(args, RunnableConfig::default())
        .await
        .expect("pipeline should succeed");

    assert_eq!(output, "langch");
}

#[tokio::test]
async fn passthrough_returns_original_input() {
    let passthrough = RunnablePassthrough::new();
    let input = String::from("value");

    let output = passthrough
        .invoke(input.clone(), RunnableConfig::default())
        .await
        .expect("passthrough should succeed");

    assert_eq!(output, input);
}

#[test]
fn human_message_exposes_role_and_content() {
    let message = BaseMessage::from(HumanMessage::new("hello"));

    assert_eq!(message.role().as_str(), "human");
    assert_eq!(message.content(), "hello");
}
