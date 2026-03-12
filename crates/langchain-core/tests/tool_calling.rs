use langchain_core::LangChainError;
use langchain_core::language_models::{
    BaseChatModel, StructuredOutput, StructuredOutputOptions, StructuredOutputSchema,
    ToolBindingOptions,
};
use langchain_core::messages::{AIMessage, BaseMessage, HumanMessage, ToolCall};
use langchain_core::output_parsers::{
    JsonOutputKeyToolsParser, JsonOutputToolsParser, parse_openai_tool_calls,
};
use langchain_core::runnables::{Runnable, RunnableConfig};
use langchain_core::tools::ToolDefinition;

#[derive(Clone)]
struct FakeToolCallingModel;

impl BaseChatModel for FakeToolCallingModel {
    fn model_name(&self) -> &str {
        "fake-tool-calling"
    }

    fn generate<'a>(
        &'a self,
        _messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<AIMessage, LangChainError>> {
        Box::pin(async move {
            Ok(AIMessage::new("").with_tool_calls(vec![ToolCall::new(
                "Answer",
                serde_json::json!({
                    "answer": "same",
                    "justification": "both weigh one pound"
                }),
            )]))
        })
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
        _options: ToolBindingOptions,
    ) -> Result<Box<dyn BaseChatModel>, LangChainError> {
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "Answer");
        Ok(Box::new(self.clone()))
    }
}

#[test]
fn parse_openai_tool_calls_splits_valid_and_invalid_entries() {
    let raw = vec![
        serde_json::json!({
            "id": "call_valid",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": "{\"query\":\"rust\"}"
            }
        }),
        serde_json::json!({
            "id": "call_invalid",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": "{not-json"
            }
        }),
    ];

    let (valid, invalid) = parse_openai_tool_calls(&raw);

    assert_eq!(valid.len(), 1);
    assert_eq!(valid[0].name(), "lookup");
    assert_eq!(valid[0].id(), Some("call_valid"));
    assert_eq!(invalid.len(), 1);
    assert_eq!(invalid[0].id(), Some("call_invalid"));
    assert!(invalid[0].error().is_some());
}

#[tokio::test]
async fn json_output_tools_parsers_extract_args_from_tool_calls() {
    let message = AIMessage::new("").with_tool_calls(vec![
        ToolCall::new("Answer", serde_json::json!({"answer": "same"})).with_id("call_1"),
        ToolCall::new("Citations", serde_json::json!({"items": ["a", "b"]})).with_id("call_2"),
    ]);

    let all_args = JsonOutputToolsParser::new()
        .invoke(message.clone(), RunnableConfig::default())
        .await
        .expect("tool parser should succeed");
    let answer_args = JsonOutputKeyToolsParser::new("Answer")
        .invoke(message, RunnableConfig::default())
        .await
        .expect("key parser should succeed");

    assert_eq!(all_args.len(), 2);
    assert_eq!(answer_args, serde_json::json!({"answer": "same"}));
}

#[tokio::test]
async fn default_with_structured_output_uses_bound_tool_parser() {
    let model = FakeToolCallingModel;
    let schema = StructuredOutputSchema::new(
        "Answer",
        serde_json::json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            },
            "required": ["answer"]
        }),
    )
    .with_description("Return the final answer");

    let runnable = model
        .with_structured_output(schema, StructuredOutputOptions::default())
        .expect("structured output should be available");
    let output = runnable
        .invoke_boxed(
            vec![BaseMessage::from(HumanMessage::new("what weighs more?"))],
            RunnableConfig::default(),
        )
        .await
        .expect("structured output invocation should succeed");

    match output {
        StructuredOutput::Parsed(value) => {
            assert_eq!(value["answer"], "same");
            assert_eq!(value["justification"], "both weigh one pound");
        }
        StructuredOutput::Raw { .. } => panic!("expected parsed-only structured output"),
    }
}
