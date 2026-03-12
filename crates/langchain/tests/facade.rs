use langchain::messages::HumanMessage;
use langchain::tools::tool;

#[test]
fn facade_reexports_messages() {
    let message = HumanMessage::new("hello");
    assert_eq!(message.content(), "hello");
}

#[test]
fn facade_reexports_tool_helper() {
    let definition = tool("lookup", "Look up a record");

    assert_eq!(definition.name(), "lookup");
    assert_eq!(definition.description(), "Look up a record");
}
