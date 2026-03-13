use langchain_classic::base_memory::BaseMemory;
use langchain_classic::memory::prompt::{ENTITY_MEMORY_CONVERSATION_TEMPLATE, SUMMARY_PROMPT};
use langchain_classic::memory::{
    ConversationBufferMemory, ConversationStringBufferMemory, get_prompt_input_key,
};
use langchain_classic::messages::{HumanMessage, messages_from_dict};
use langchain_classic::prompts::{
    ChatPromptTemplate, Prompt, PromptArgument, PromptMessageTemplate,
};
use serde_json::json;

#[test]
fn classic_prompts_module_exposes_prompt_alias_and_chat_surface() {
    let alias = Prompt::new("Hello {name}");
    let rendered = alias
        .format(&[("name".to_owned(), PromptArgument::String("Rust".to_owned()))].into())
        .expect("classic prompts alias should render via the shared prompt template");
    assert_eq!(rendered, "Hello Rust");

    let prompt = ChatPromptTemplate::from_messages([
        PromptMessageTemplate::system("You are concise."),
        PromptMessageTemplate::placeholder("history"),
        PromptMessageTemplate::human("{input}"),
    ]);
    let rendered_messages = prompt
        .format_messages(
            &[
                (
                    "history".to_owned(),
                    PromptArgument::Messages(vec![HumanMessage::new("Earlier turn").into()]),
                ),
                (
                    "input".to_owned(),
                    PromptArgument::String("Latest turn".to_owned()),
                ),
            ]
            .into(),
        )
        .expect("classic prompts module should keep the shared chat prompt helpers");

    assert_eq!(rendered_messages.len(), 3);
    assert_eq!(rendered_messages[0].content(), "You are concise.");
    assert_eq!(rendered_messages[1].content(), "Earlier turn");
    assert_eq!(rendered_messages[2].content(), "Latest turn");
}

#[test]
fn classic_conversation_buffer_memory_supports_string_and_message_views() {
    let memory = ConversationBufferMemory::new();
    memory.save_context(
        [("input".to_owned(), json!("hello"))].into(),
        [("output".to_owned(), json!("world"))].into(),
    );

    assert_eq!(memory.memory_variables(), vec!["history".to_owned()]);
    assert_eq!(
        memory
            .load_memory_variables(Default::default())
            .get("history"),
        Some(&json!("Human: hello\nAI: world"))
    );
    assert_eq!(memory.buffer_as_messages().len(), 2);

    let message_memory = ConversationBufferMemory::new().with_return_messages(true);
    message_memory.save_context(
        [("input".to_owned(), json!("hi again"))].into(),
        [("output".to_owned(), json!("welcome back"))].into(),
    );

    let loaded = message_memory.load_memory_variables(Default::default());
    let history = loaded["history"]
        .as_array()
        .expect("return_messages memory should serialize to a JSON array");
    let messages = messages_from_dict(history).expect("message array should deserialize");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].content(), "hi again");
    assert_eq!(messages[1].content(), "welcome back");
    assert_eq!(
        message_memory
            .buffer_as_str()
            .expect("string buffer should still be derivable"),
        "Human: hi again\nAI: welcome back"
    );
}

#[test]
fn classic_conversation_string_buffer_memory_uses_prompt_keys() {
    let inferred = get_prompt_input_key(
        &[
            ("history".to_owned(), json!("existing")),
            ("question".to_owned(), json!("Where are we?")),
            ("stop".to_owned(), json!(["."])),
        ]
        .into(),
        &["history".to_owned()],
    )
    .expect("helper should ignore memory variables and stop");
    assert_eq!(inferred, "question");

    let memory = ConversationStringBufferMemory::new()
        .with_input_key("question")
        .with_output_key("answer");
    memory.save_context(
        [("question".to_owned(), json!("Where are we?"))].into(),
        [
            ("answer".to_owned(), json!("In tests")),
            ("metadata".to_owned(), json!("ignored")),
        ]
        .into(),
    );

    assert_eq!(memory.memory_variables(), vec!["history".to_owned()]);
    assert_eq!(
        memory
            .load_memory_variables(Default::default())
            .get("history"),
        Some(&json!("\nHuman: Where are we?\nAI: In tests"))
    );
    memory.clear();
    assert_eq!(
        memory
            .load_memory_variables(Default::default())
            .get("history"),
        Some(&json!(""))
    );
}

#[test]
fn classic_memory_prompt_templates_render_from_shared_prompt_template() {
    let summary = SUMMARY_PROMPT
        .format(
            &[
                (
                    "summary".to_owned(),
                    PromptArgument::String("Old summary".to_owned()),
                ),
                (
                    "new_lines".to_owned(),
                    PromptArgument::String("Human: hi\nAI: hello".to_owned()),
                ),
            ]
            .into(),
        )
        .expect("summary prompt should render");
    assert!(summary.contains("Old summary"));
    assert!(summary.contains("Human: hi"));

    let entity_prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE
        .format(
            &[
                (
                    "entities".to_owned(),
                    PromptArgument::String("LangChain".to_owned()),
                ),
                (
                    "history".to_owned(),
                    PromptArgument::String("Human: hi".to_owned()),
                ),
                (
                    "input".to_owned(),
                    PromptArgument::String("Tell me more".to_owned()),
                ),
            ]
            .into(),
        )
        .expect("entity memory conversation prompt should render");
    assert!(entity_prompt.contains("Context:\nLangChain"));
    assert!(entity_prompt.contains("Human: Tell me more"));
}
