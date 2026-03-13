use langchain_core::messages::ToolCall;
use langchain_core::messages::ToolMessageStatus;
use langchain_core::tools::BaseTool;

pub trait ToolUnitHarness {
    type Tool: BaseTool;

    fn tool(&self) -> Self::Tool;

    fn example_tool_call(&self) -> ToolCall;
}

pub struct ToolsUnitTests<H> {
    harness: H,
}

impl<H> ToolsUnitTests<H> {
    pub fn new(harness: H) -> Self {
        Self { harness }
    }
}

impl<H> ToolsUnitTests<H>
where
    H: ToolUnitHarness,
{
    pub async fn run(&self) {
        let tool = self.harness.tool();
        let definition = tool.definition();
        assert!(!definition.name().is_empty());
        assert!(!definition.description().is_empty());
        assert!(definition.parameters().is_object());

        let call = self.harness.example_tool_call();
        let message = tool
            .invoke(call.clone(), Default::default())
            .await
            .expect("tool unit harness should invoke the tool");
        assert_eq!(message.status(), ToolMessageStatus::Success);
        assert_eq!(message.tool_call_id(), call.id().unwrap_or_default());
        assert!(!message.content().is_empty());
    }
}

impl<H> ToolUnitHarness for &H
where
    H: ToolUnitHarness,
{
    type Tool = H::Tool;

    fn tool(&self) -> Self::Tool {
        (*self).tool()
    }

    fn example_tool_call(&self) -> ToolCall {
        (*self).example_tool_call()
    }
}
