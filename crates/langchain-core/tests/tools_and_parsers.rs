use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::documents::Document;
use langchain_core::messages::{AIMessage, ToolCall};
use langchain_core::output_parsers::{
    CommaSeparatedListOutputParser, JsonOutputParser, MarkdownListOutputParser,
    NumberedListOutputParser, PydanticOutputParser, PydanticToolsParser, SimpleJsonOutputParser,
    XMLOutputParser,
};
use langchain_core::retrievers::BaseRetriever;
use langchain_core::runnables::{Runnable, RunnableConfig, RunnableLambda};
use langchain_core::tools::{
    BaseTool, BaseToolkit, RetrieverInput, SchemaAnnotationError, Tool, convert_runnable_to_tool,
    create_retriever_tool, render_text_description, render_text_description_and_args, tool,
};
use langchain_core::vectorstores::VectorStoreRetriever;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone)]
struct StaticRetriever;

impl BaseRetriever for StaticRetriever {
    fn get_relevant_documents<'a>(
        &'a self,
        query: &'a str,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        Box::pin(async move { Ok(vec![Document::new(format!("match:{query}"))]) })
    }
}

#[derive(Clone)]
struct ToolkitFixture;

impl BaseToolkit for ToolkitFixture {
    fn tools(&self) -> Vec<Box<dyn BaseTool>> {
        vec![Box::new(Tool::new(
            tool("echo", "Echo a string"),
            |input| Box::pin(async move { Ok(input) }),
        ))]
    }
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct AnswerPayload {
    answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct MultiplyInput {
    a: i32,
    b: Vec<i32>,
}

#[tokio::test]
async fn tool_helpers_render_and_invoke() {
    let toolkit = ToolkitFixture;
    let tools = toolkit.tools();
    let tool_refs = tools
        .iter()
        .map(|tool| tool.as_ref() as &dyn BaseTool)
        .collect::<Vec<_>>();

    let rendered = render_text_description(&tool_refs);
    let rendered_with_args = render_text_description_and_args(&tool_refs);

    assert!(rendered.contains("echo: Echo a string"));
    assert!(rendered_with_args.contains("\"type\":\"object\""));

    let response = tools[0]
        .invoke(
            ToolCall::new("echo", json!({ "input": "hello" })).with_id("call_1"),
            RunnableConfig::default(),
        )
        .await
        .expect("tool invocation should succeed");

    assert_eq!(response.content(), "hello");
    assert_eq!(response.name(), Some("echo"));
}

#[tokio::test]
async fn retriever_tool_serializes_documents_and_vectorstore_reexport_exists() {
    let retriever_tool = create_retriever_tool(StaticRetriever, "lookup", "Look up docs");
    let message = retriever_tool
        .invoke(
            ToolCall::new("lookup", json!({ "query": "rust" })).with_id("call_docs"),
            RunnableConfig::default(),
        )
        .await
        .expect("retriever tool should succeed");

    assert!(message.content().contains("match:rust"));

    let _ = VectorStoreRetriever::new(langchain_core::vectorstores::InMemoryVectorStore::new(
        langchain_core::embeddings::CharacterEmbeddings::new(),
    ));
}

#[test]
fn retriever_input_public_surface_round_trips_and_matches_tool_schema() {
    let input = RetrieverInput::new("rust");

    assert_eq!(
        serde_json::to_value(&input).expect("serialize retriever input"),
        json!({ "query": "rust" })
    );
    assert_eq!(
        serde_json::from_value::<RetrieverInput>(json!({ "query": "langchain" }))
            .expect("deserialize retriever input"),
        RetrieverInput::new("langchain")
    );
    assert_eq!(
        RetrieverInput::json_schema(),
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "query to look up in retriever"
                }
            },
            "required": ["query"]
        })
    );

    let retriever_tool = create_retriever_tool(StaticRetriever, "lookup", "Look up docs");
    assert_eq!(
        retriever_tool.definition().parameters(),
        &RetrieverInput::json_schema()
    );
}

#[tokio::test]
async fn convert_runnable_to_tool_handles_string_and_structured_inputs() {
    let string_tool = convert_runnable_to_tool(
        RunnableLambda::new(|input: String| async move { Ok(format!("{input}z")) }),
        tool("append_z", "Append a suffix"),
    );

    assert_eq!(
        string_tool.definition().parameters(),
        &json!({
            "type": "object",
            "properties": {
                "input": { "type": "string" }
            },
            "required": ["input"]
        })
    );

    let string_message = string_tool
        .invoke(
            ToolCall::new("append_z", json!({ "input": "ba" })).with_id("call_suffix"),
            RunnableConfig::default(),
        )
        .await
        .expect("string runnable should convert to a tool");
    assert_eq!(string_message.content(), "baz");
    assert_eq!(string_message.artifact(), Some(&json!("baz")));

    let structured_tool = convert_runnable_to_tool(
        RunnableLambda::new(|input: MultiplyInput| async move {
            Ok(input.a * input.b.into_iter().max().unwrap_or_default())
        }),
        tool("multiply", "Multiply by the largest entry").with_parameters(json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer" },
                "b": {
                    "type": "array",
                    "items": { "type": "integer" }
                }
            },
            "required": ["a", "b"]
        })),
    );

    let structured_message = structured_tool
        .invoke(
            ToolCall::new("multiply", json!({ "a": 3, "b": [1, 2] })).with_id("call_product"),
            RunnableConfig::default(),
        )
        .await
        .expect("structured runnable should convert to a tool");
    assert_eq!(structured_message.content(), "6");
    assert_eq!(structured_message.artifact(), Some(&json!(6)));
}

#[test]
fn schema_annotation_error_is_displayable() {
    let error = SchemaAnnotationError::new("missing schema");
    assert_eq!(error.to_string(), "missing schema");
}

#[tokio::test]
async fn json_and_list_parsers_cover_minimal_public_surface() {
    let json_message = AIMessage::new(r#"{"answer":"same"}"#);
    let json_value = JsonOutputParser::new()
        .invoke(json_message.clone(), RunnableConfig::default())
        .await
        .expect("json parser should succeed");
    let simple_value = SimpleJsonOutputParser::new()
        .invoke(json_message, RunnableConfig::default())
        .await
        .expect("simple json parser should succeed");

    assert_eq!(json_value, simple_value);

    let markdown_items = MarkdownListOutputParser::new()
        .invoke(AIMessage::new("- alpha\n- beta"), RunnableConfig::default())
        .await
        .expect("markdown list parser should succeed");
    let numbered_items = NumberedListOutputParser::new()
        .invoke(
            AIMessage::new("1. first\n2. second"),
            RunnableConfig::default(),
        )
        .await
        .expect("numbered list parser should succeed");
    let comma_items = CommaSeparatedListOutputParser::new()
        .invoke(
            AIMessage::new("red, green, blue"),
            RunnableConfig::default(),
        )
        .await
        .expect("comma list parser should succeed");

    assert_eq!(markdown_items, vec!["alpha", "beta"]);
    assert_eq!(numbered_items, vec!["first", "second"]);
    assert_eq!(comma_items, vec!["red", "green", "blue"]);
}

#[tokio::test]
async fn pydantic_style_parsers_deserialize_structured_payloads() {
    let parsed = PydanticOutputParser::<AnswerPayload>::new()
        .invoke(
            AIMessage::new(r#"{"answer":"Boston"}"#),
            RunnableConfig::default(),
        )
        .await
        .expect("pydantic-style json parser should succeed");

    assert_eq!(
        parsed,
        AnswerPayload {
            answer: "Boston".to_owned(),
        }
    );

    let tool_payloads = PydanticToolsParser::<AnswerPayload>::new()
        .invoke(
            AIMessage::new("").with_tool_calls(vec![
                ToolCall::new("AnswerPayload", json!({ "answer": "Paris" })),
                ToolCall::new("AnswerPayload", json!({ "answer": "Tokyo" })),
            ]),
            RunnableConfig::default(),
        )
        .await
        .expect("pydantic-style tool parser should succeed");

    assert_eq!(
        tool_payloads,
        vec![
            AnswerPayload {
                answer: "Paris".to_owned(),
            },
            AnswerPayload {
                answer: "Tokyo".to_owned(),
            },
        ]
    );
}

#[tokio::test]
async fn xml_output_parser_handles_simple_nested_documents() {
    let parsed = XMLOutputParser::new()
        .invoke(
            AIMessage::new("<response><answer>42</answer><items><item>a</item><item>b</item></items></response>"),
            RunnableConfig::default(),
        )
        .await
        .expect("xml parser should succeed");

    assert_eq!(
        parsed,
        json!({
            "response": {
                "answer": "42",
                "items": {
                    "item": ["a", "b"]
                }
            }
        })
    );
}
