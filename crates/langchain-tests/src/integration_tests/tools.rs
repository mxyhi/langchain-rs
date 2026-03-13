use langchain_core::messages::ToolMessageStatus;
use langchain_core::tools::BaseTool;

use crate::{assert_tool_invocation, unit_tests::ToolUnitHarness, unit_tests::ToolsUnitTests};

pub trait ToolIntegrationHarness: ToolUnitHarness {
    fn expected_content(&self) -> &'static str {
        "result:rust"
    }
}

pub struct ToolsIntegrationTests<H> {
    harness: H,
}

impl<H> ToolsIntegrationTests<H> {
    pub fn new(harness: H) -> Self {
        Self { harness }
    }
}

impl<H> ToolsIntegrationTests<H>
where
    H: ToolIntegrationHarness,
{
    pub async fn run(&self) {
        ToolsUnitTests::new(&self.harness).run().await;

        let tool = self.harness.tool();
        let call = self.harness.example_tool_call();
        assert_tool_invocation(&tool, call.clone(), self.harness.expected_content())
            .await
            .expect("tool integration invocation should succeed");

        let message = tool
            .invoke(call, Default::default())
            .await
            .expect("tool integration should return tool message");
        assert_eq!(message.tool_call_id(), "call_lookup_1");
        assert_eq!(message.status(), ToolMessageStatus::Success);
        assert_eq!(message.name(), Some("lookup"));
    }
}
