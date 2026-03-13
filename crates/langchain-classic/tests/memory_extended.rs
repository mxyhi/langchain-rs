use langchain_classic::base_memory::BaseMemory;
use langchain_classic::memory::{
    BaseChatMessageHistory, CombinedMemory, ConversationBufferMemory,
    ConversationBufferWindowMemory, ConversationStringBufferMemory, ReadOnlySharedMemory,
    SimpleMemory,
};
use langchain_classic::messages::BaseMessage;
use serde_json::json;
use std::sync::RwLock;

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

#[test]
fn conversation_buffer_memory_supports_custom_prefix_and_buffer_views() {
    let memory = ConversationBufferMemory::new()
        .with_memory_key("foo")
        .with_human_prefix("Friend")
        .with_ai_prefix("Assistant");

    memory.save_context(
        [("input".to_owned(), json!("bar"))].into(),
        [("output".to_owned(), json!("baz"))].into(),
    );

    assert_eq!(
        memory.load_memory_variables(Default::default()),
        [("foo".to_owned(), json!("Friend: bar\nAssistant: baz"))].into()
    );
    assert_eq!(
        memory.buffer().as_text(),
        Some("Friend: bar\nAssistant: baz")
    );

    let message_mode = ConversationBufferMemory::new()
        .with_return_messages(true)
        .with_human_prefix("Friend")
        .with_ai_prefix("Assistant");
    message_mode.save_context(
        [("input".to_owned(), json!("left"))].into(),
        [("output".to_owned(), json!("right"))].into(),
    );

    let messages = message_mode
        .buffer()
        .into_messages()
        .expect("message mode should expose message buffers");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].content(), "left");
    assert_eq!(messages[1].content(), "right");
}

#[tokio::test]
async fn conversation_buffer_memory_can_use_custom_chat_history_async_surface() {
    let memory = ConversationBufferMemory::new()
        .with_memory_key("foo")
        .with_ai_prefix("Assistant")
        .with_chat_memory(AsyncOnlyHistory::default());

    memory
        .asave_context(
            [("input".to_owned(), json!("bar"))].into(),
            [("output".to_owned(), json!("baz"))].into(),
        )
        .await;

    assert_eq!(
        memory.aload_memory_variables(Default::default()).await,
        [("foo".to_owned(), json!("Human: bar\nAssistant: baz"))].into()
    );

    memory.aclear().await;
    assert_eq!(
        memory.aload_memory_variables(Default::default()).await,
        [("foo".to_owned(), json!(""))].into()
    );
}

#[test]
fn buffer_window_memory_supports_custom_prefixes_and_zero_window() {
    let empty = ConversationBufferWindowMemory::new()
        .with_memory_key("foo")
        .with_human_prefix("Friend")
        .with_ai_prefix("Assistant")
        .with_k(0);

    empty.save_context(
        [("input".to_owned(), json!("bar"))].into(),
        [("output".to_owned(), json!("baz"))].into(),
    );
    assert_eq!(
        empty.load_memory_variables(Default::default()),
        [("foo".to_owned(), json!(""))].into()
    );
    assert_eq!(empty.buffer().as_text(), Some(""));

    let memory = ConversationBufferWindowMemory::new()
        .with_memory_key("foo")
        .with_human_prefix("Friend")
        .with_ai_prefix("Assistant")
        .with_k(1);
    memory.save_context(
        [("input".to_owned(), json!("bar"))].into(),
        [("output".to_owned(), json!("baz"))].into(),
    );
    assert_eq!(
        memory.load_memory_variables(Default::default()),
        [("foo".to_owned(), json!("Friend: bar\nAssistant: baz"))].into()
    );
}

#[test]
fn readonly_shared_memory_clear_is_local_noop() {
    let source = ConversationBufferMemory::new().with_memory_key("baz");
    source.save_context(
        [("input".to_owned(), json!("persist"))].into(),
        [("output".to_owned(), json!("value"))].into(),
    );

    let memory = ReadOnlySharedMemory::new(source);
    memory.clear();

    assert_eq!(
        memory.load_memory_variables(Default::default()),
        [("baz".to_owned(), json!("Human: persist\nAI: value"))].into()
    );
}

#[test]
fn combined_memory_clear_resets_all_children() {
    let foo = ConversationBufferMemory::new().with_memory_key("foo");
    let bar = ConversationBufferMemory::new().with_memory_key("bar");
    let memory = CombinedMemory::new(vec![Box::new(foo), Box::new(bar)])
        .expect("combined memory with distinct keys should construct");

    memory.save_context(
        [("input".to_owned(), json!("Hello there"))].into(),
        [("output".to_owned(), json!("Hello, how can I help you?"))].into(),
    );
    memory.clear();

    assert_eq!(
        memory.load_memory_variables(Default::default()),
        [("foo".to_owned(), json!("")), ("bar".to_owned(), json!(""))].into()
    );
}

#[test]
fn conversation_string_buffer_memory_matches_python_newline_accumulation() {
    let memory = ConversationStringBufferMemory::new()
        .with_input_key("question")
        .with_output_key("answer");

    memory.save_context(
        [("question".to_owned(), json!("Where are we?"))].into(),
        [("answer".to_owned(), json!("In tests"))].into(),
    );

    assert_eq!(
        memory.load_memory_variables(Default::default()),
        [(
            "history".to_owned(),
            json!("\nHuman: Where are we?\nAI: In tests")
        )]
        .into()
    );
}

#[derive(Default)]
struct AsyncOnlyHistory {
    messages: RwLock<Vec<BaseMessage>>,
}

impl BaseChatMessageHistory for AsyncOnlyHistory {
    fn messages(&self) -> Vec<BaseMessage> {
        Vec::new()
    }

    fn add_message(&self, _message: BaseMessage) {}

    fn clear(&self) {}

    fn aget_messages<'a>(&'a self) -> futures_util::future::BoxFuture<'a, Vec<BaseMessage>> {
        Box::pin(async move {
            self.messages
                .read()
                .expect("async-only history read lock poisoned")
                .clone()
        })
    }

    fn aadd_messages<'a>(
        &'a self,
        messages: Vec<BaseMessage>,
    ) -> futures_util::future::BoxFuture<'a, ()> {
        Box::pin(async move {
            self.messages
                .write()
                .expect("async-only history write lock poisoned")
                .extend(messages);
        })
    }

    fn aclear<'a>(&'a self) -> futures_util::future::BoxFuture<'a, ()> {
        Box::pin(async move {
            self.messages
                .write()
                .expect("async-only history write lock poisoned")
                .clear();
        })
    }
}
