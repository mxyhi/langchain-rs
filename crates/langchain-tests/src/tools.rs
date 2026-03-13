use langchain_core::LangChainError;
use langchain_core::messages::{ToolCall, ToolMessageStatus};
use langchain_core::runnables::RunnableConfig;
use langchain_core::tools::BaseTool;

pub async fn assert_tool_invocation<T>(
    tool: &T,
    input: ToolCall,
    expected_content: &str,
) -> Result<(), LangChainError>
where
    T: BaseTool,
{
    let message = tool.invoke(input, RunnableConfig::default()).await?;

    assert_eq!(message.content(), expected_content);
    assert_eq!(message.status(), ToolMessageStatus::Success);

    Ok(())
}

pub async fn assert_tool_round_trip<T>(
    tool: &T,
    input: ToolCall,
    expected_name: &str,
    expected_content: &str,
) where
    T: BaseTool,
{
    let message = tool
        .invoke(input, RunnableConfig::default())
        .await
        .expect("tool round trip should succeed");
    assert_eq!(message.name(), Some(expected_name));
    assert_eq!(message.content(), expected_content);
    assert_eq!(message.status(), ToolMessageStatus::Success);
}

pub async fn assert_structured_tool<T>(
    tool: &T,
    input: ToolCall,
    expected_name: &str,
    expected_artifact: serde_json::Value,
) where
    T: BaseTool,
{
    let message = tool
        .invoke(input, RunnableConfig::default())
        .await
        .expect("structured tool should succeed");
    assert_eq!(message.name(), Some(expected_name));
    assert_eq!(message.artifact(), Some(&expected_artifact));
    assert_eq!(message.status(), ToolMessageStatus::Success);
}
