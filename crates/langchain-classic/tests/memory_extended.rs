use langchain_classic::base_memory::BaseMemory;
use langchain_classic::memory::{
    CombinedMemory, ConversationBufferMemory, ConversationBufferWindowMemory, ReadOnlySharedMemory,
    SimpleMemory,
};
use serde_json::json;

#[test]
fn simple_memory_returns_seeded_values_without_mutation() {
    let memory = SimpleMemory::new([("baz", json!("foo")), ("count", json!(2))]);

    assert_eq!(
        memory.memory_variables(),
        vec!["baz".to_owned(), "count".to_owned()]
    );
    assert_eq!(
        memory.load_memory_variables(Default::default()),
        [
            ("baz".to_owned(), json!("foo")),
            ("count".to_owned(), json!(2))
        ]
        .into()
    );

    memory.save_context(
        [("input".to_owned(), json!("bar"))].into(),
        [("output".to_owned(), json!("qux"))].into(),
    );
    assert_eq!(
        memory.load_memory_variables(Default::default()),
        [
            ("baz".to_owned(), json!("foo")),
            ("count".to_owned(), json!(2))
        ]
        .into()
    );
}

#[test]
fn readonly_shared_memory_reflects_source_but_ignores_local_writes() {
    let source = ConversationBufferMemory::new().with_memory_key("baz");
    let memory = ReadOnlySharedMemory::new(source);

    memory.save_context(
        [("input".to_owned(), json!("ignored"))].into(),
        [("output".to_owned(), json!("ignored"))].into(),
    );
    assert_eq!(
        memory.load_memory_variables(Default::default()),
        [("baz".to_owned(), json!(""))].into()
    );

    memory.inner().save_context(
        [("input".to_owned(), json!("bar"))].into(),
        [("output".to_owned(), json!("foo"))].into(),
    );
    assert_eq!(
        memory.load_memory_variables(Default::default()),
        [("baz".to_owned(), json!("Human: bar\nAI: foo"))].into()
    );
}

#[test]
fn buffer_window_memory_keeps_only_last_k_turns() {
    let memory = ConversationBufferWindowMemory::new()
        .with_memory_key("baz")
        .with_k(1);

    memory.save_context(
        [("input".to_owned(), json!("first"))].into(),
        [("output".to_owned(), json!("one"))].into(),
    );
    memory.save_context(
        [("input".to_owned(), json!("second"))].into(),
        [("output".to_owned(), json!("two"))].into(),
    );

    assert_eq!(memory.memory_variables(), vec!["baz".to_owned()]);
    assert_eq!(
        memory.load_memory_variables(Default::default()),
        [("baz".to_owned(), json!("Human: second\nAI: two"))].into()
    );

    let messages = ConversationBufferWindowMemory::new()
        .with_k(1)
        .with_return_messages(true);
    messages.save_context(
        [("input".to_owned(), json!("first"))].into(),
        [("output".to_owned(), json!("one"))].into(),
    );
    messages.save_context(
        [("input".to_owned(), json!("second"))].into(),
        [("output".to_owned(), json!("two"))].into(),
    );
    let loaded = messages.load_memory_variables(Default::default());
    let history = loaded["history"]
        .as_array()
        .expect("message mode should serialize as an array");
    assert_eq!(history.len(), 2);
}

#[test]
fn combined_memory_merges_unique_memory_variables_and_delegates_writes() {
    let foo = ConversationBufferMemory::new().with_memory_key("foo");
    let bar = ConversationBufferMemory::new().with_memory_key("bar");
    let memory = CombinedMemory::new(vec![Box::new(foo), Box::new(bar)])
        .expect("combined memory with distinct keys should construct");

    assert_eq!(
        memory.memory_variables(),
        vec!["foo".to_owned(), "bar".to_owned()]
    );
    assert_eq!(
        memory.load_memory_variables(Default::default()),
        [("foo".to_owned(), json!("")), ("bar".to_owned(), json!(""))].into()
    );

    memory.save_context(
        [("input".to_owned(), json!("Hello there"))].into(),
        [("output".to_owned(), json!("Hello, how can I help you?"))].into(),
    );

    assert_eq!(
        memory.load_memory_variables(Default::default()),
        [
            (
                "foo".to_owned(),
                json!("Human: Hello there\nAI: Hello, how can I help you?")
            ),
            (
                "bar".to_owned(),
                json!("Human: Hello there\nAI: Hello, how can I help you?")
            ),
        ]
        .into()
    );
}

#[test]
fn combined_memory_rejects_repeated_memory_variables() {
    let duplicate_a = ConversationBufferMemory::new().with_memory_key("bar");
    let duplicate_b = ConversationBufferMemory::new().with_memory_key("bar");

    let error = CombinedMemory::new(vec![Box::new(duplicate_a), Box::new(duplicate_b)])
        .expect_err("duplicate memory keys should be rejected");

    assert!(
        error
            .to_string()
            .contains("The same variables {\"bar\"} are found in multiple memory objects"),
        "unexpected error: {error}"
    );
}
