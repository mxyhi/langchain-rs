use futures_util::future::BoxFuture;
use langchain::agents::{AgentState, create_agent};
use langchain::language_models::{
    BaseChatModel, StructuredOutput, StructuredOutputOptions, StructuredOutputSchema,
    ToolBindingOptions,
};
use langchain::messages::{AIMessage, BaseMessage, HumanMessage, ToolCall};
use langchain::runnables::{Runnable, RunnableConfig, RunnableDyn};
use langchain::tools::ToolDefinition;
use serde_json::json;

#[tokio::test]
async fn create_agent_builds_state_with_system_and_human_messages() {
    let agent = create_agent(langchain::language_models::ParrotChatModel::new(
        "parrot-agent",
        24,
    ))
    .with_system_prompt("You are concise.");

    let state = agent
        .invoke("hello".to_owned(), Default::default())
        .await
        .expect("agent should succeed");

    assert_eq!(state.messages().len(), 3);
    assert_eq!(
        state.messages()[0],
        BaseMessage::from(langchain::messages::SystemMessage::new("You are concise."))
    );
    assert_eq!(
        state.messages()[1],
        BaseMessage::from(HumanMessage::new("hello"))
    );
    assert_eq!(state.messages()[2].content(), "hello");
    assert_eq!(state.structured_response(), None);
}

#[tokio::test]
async fn create_agent_can_capture_structured_output() {
    let agent =
        create_agent(FakeStructuredOutputModel).with_response_format(StructuredOutputSchema::new(
            "AnswerPayload",
            json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                },
                "required": ["answer"]
            }),
        ));

    let state = agent
        .invoke("ping".to_owned(), Default::default())
        .await
        .expect("structured agent should succeed");

    assert_eq!(
        state.structured_response(),
        Some(&json!({ "answer": "pong" }))
    );
    assert!(matches!(state, AgentState { .. }));
}

#[tokio::test]
async fn create_agent_rejects_multiple_structured_outputs() {
    let agent = create_agent(FakeMultipleStructuredOutputModel).with_response_format(
        StructuredOutputSchema::new(
            "AnswerPayload",
            json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                },
                "required": ["answer"]
            }),
        ),
    );

    let error = agent
        .invoke("ping".to_owned(), Default::default())
        .await
        .expect_err("multiple structured outputs should fail");

    assert!(
        error
            .to_string()
            .contains("multiple structured responses (AnswerPayload, AnswerPayload)"),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn create_agent_surfaces_structured_output_validation_error() {
    let agent = create_agent(FakeStructuredOutputValidationModel).with_response_format(
        StructuredOutputSchema::new(
            "AnswerPayload",
            json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                },
                "required": ["answer"]
            }),
        ),
    );

    let error = agent
        .invoke("ping".to_owned(), Default::default())
        .await
        .expect_err("parsing failures should surface a structured output validation error");

    assert!(
        error.to_string().contains(
            "Failed to parse structured output for tool 'AnswerPayload': invalid JSON payload."
        ),
        "unexpected error: {error}"
    );
}

#[derive(Clone)]
struct FakeStructuredOutputModel;

impl BaseChatModel for FakeStructuredOutputModel {
    fn model_name(&self) -> &str {
        "fake-structured-model"
    }

    fn generate<'a>(
        &'a self,
        _messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, langchain::LangChainError>> {
        Box::pin(async move {
            Ok(AIMessage::new("").with_tool_calls(vec![ToolCall::new(
                "AnswerPayload",
                json!({ "answer": "pong" }),
            )]))
        })
    }

    fn bind_tools(
        &self,
        _tools: Vec<ToolDefinition>,
        _options: ToolBindingOptions,
    ) -> Result<Box<dyn BaseChatModel>, langchain::LangChainError> {
        Ok(Box::new(self.clone()))
    }

    fn with_structured_output(
        &self,
        _schema: StructuredOutputSchema,
        _options: StructuredOutputOptions,
    ) -> Result<Box<dyn RunnableDyn<Vec<BaseMessage>, StructuredOutput>>, langchain::LangChainError>
    {
        Ok(Box::new(FakeStructuredOutputRunnable))
    }
}

struct FakeStructuredOutputRunnable;

impl Runnable<Vec<BaseMessage>, StructuredOutput> for FakeStructuredOutputRunnable {
    fn invoke<'a>(
        &'a self,
        _input: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<StructuredOutput, langchain::LangChainError>> {
        Box::pin(async move {
            Ok(StructuredOutput::Raw {
                raw: AIMessage::new("").with_tool_calls(vec![ToolCall::new(
                    "AnswerPayload",
                    json!({ "answer": "pong" }),
                )]),
                parsed: Some(json!({ "answer": "pong" })),
                parsing_error: None,
            })
        })
    }
}

#[derive(Clone)]
struct FakeStructuredOutputValidationModel;

impl BaseChatModel for FakeStructuredOutputValidationModel {
    fn model_name(&self) -> &str {
        "fake-structured-validation-model"
    }

    fn generate<'a>(
        &'a self,
        _messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, langchain::LangChainError>> {
        Box::pin(async move { Ok(AIMessage::new("")) })
    }

    fn bind_tools(
        &self,
        _tools: Vec<ToolDefinition>,
        _options: ToolBindingOptions,
    ) -> Result<Box<dyn BaseChatModel>, langchain::LangChainError> {
        Ok(Box::new(self.clone()))
    }

    fn with_structured_output(
        &self,
        _schema: StructuredOutputSchema,
        _options: StructuredOutputOptions,
    ) -> Result<Box<dyn RunnableDyn<Vec<BaseMessage>, StructuredOutput>>, langchain::LangChainError>
    {
        Ok(Box::new(FakeStructuredOutputValidationRunnable))
    }
}

struct FakeStructuredOutputValidationRunnable;

impl Runnable<Vec<BaseMessage>, StructuredOutput> for FakeStructuredOutputValidationRunnable {
    fn invoke<'a>(
        &'a self,
        _input: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<StructuredOutput, langchain::LangChainError>> {
        Box::pin(async move {
            Ok(StructuredOutput::Raw {
                raw: AIMessage::new("").with_tool_calls(vec![ToolCall::new(
                    "AnswerPayload",
                    json!({ "answer": "pong" }),
                )]),
                parsed: None,
                parsing_error: Some("invalid JSON payload.".to_owned()),
            })
        })
    }
}

#[derive(Clone)]
struct FakeMultipleStructuredOutputModel;

impl BaseChatModel for FakeMultipleStructuredOutputModel {
    fn model_name(&self) -> &str {
        "fake-multiple-structured-model"
    }

    fn generate<'a>(
        &'a self,
        _messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, langchain::LangChainError>> {
        Box::pin(async move { Ok(AIMessage::new("")) })
    }

    fn bind_tools(
        &self,
        _tools: Vec<ToolDefinition>,
        _options: ToolBindingOptions,
    ) -> Result<Box<dyn BaseChatModel>, langchain::LangChainError> {
        Ok(Box::new(self.clone()))
    }

    fn with_structured_output(
        &self,
        _schema: StructuredOutputSchema,
        _options: StructuredOutputOptions,
    ) -> Result<Box<dyn RunnableDyn<Vec<BaseMessage>, StructuredOutput>>, langchain::LangChainError>
    {
        Ok(Box::new(FakeMultipleStructuredOutputRunnable))
    }
}

struct FakeMultipleStructuredOutputRunnable;

impl Runnable<Vec<BaseMessage>, StructuredOutput> for FakeMultipleStructuredOutputRunnable {
    fn invoke<'a>(
        &'a self,
        _input: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<StructuredOutput, langchain::LangChainError>> {
        Box::pin(async move {
            Ok(StructuredOutput::Raw {
                raw: AIMessage::new("").with_tool_calls(vec![
                    ToolCall::new("AnswerPayload", json!({ "answer": "pong" })),
                    ToolCall::new("AnswerPayload", json!({ "answer": "still pong" })),
                ]),
                parsed: Some(json!({ "answer": "pong" })),
                parsing_error: None,
            })
        })
    }
}
