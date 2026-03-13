use std::sync::Arc;

use langchain_classic::base_language::BaseLanguageModel;
use langchain_classic::base_memory::BaseMemory;
use langchain_classic::cache::InMemoryCache;
use langchain_classic::callbacks::{
    CallbackEvent, CallbackEventKind, CallbackRun, CallbackRunConfig,
};
use langchain_classic::chat_loaders::BaseChatLoader;
use langchain_classic::document_loaders::BaseLoader;
use langchain_classic::env::get_runtime_environment;
use langchain_classic::formatting::{StrictFormatter, formatter};
use langchain_classic::globals::{
    get_debug, get_llm_cache, get_verbose, set_debug, set_llm_cache, set_verbose,
};
use langchain_classic::indexing::{InMemoryRecordManager, index};
use langchain_classic::input::{get_bolded_text, get_color_mapping, get_colored_text};
use langchain_classic::load::{Reviver, SerializedValue, dumpd};
use langchain_classic::messages::HumanMessage;
use langchain_classic::prompts::PromptArgument;
use langchain_classic::storage::{BaseStore, InMemoryStore};
use langchain_classic::text_splitter::{
    CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter, TokenTextSplitter,
    Tokenizer, split_text_on_tokens,
};
use langchain_classic::utils::{comma_list, get_from_env, stringify_dict};
use langchain_classic::{Prompt, PromptTemplate};
use langchain_core::chat_sessions::ChatSession;
use langchain_core::documents::Document;
use langchain_core::embeddings::CharacterEmbeddings;
use langchain_core::vectorstores::{InMemoryVectorStore, VectorStore};
use serde_json::{Value, json};

#[derive(Default)]
struct RecorderMemory {
    values: std::sync::Mutex<std::collections::BTreeMap<String, Value>>,
}

impl BaseMemory for RecorderMemory {
    fn memory_variables(&self) -> Vec<String> {
        self.values
            .lock()
            .expect("memory lock")
            .keys()
            .cloned()
            .collect()
    }

    fn load_memory_variables(
        &self,
        _inputs: std::collections::BTreeMap<String, Value>,
    ) -> std::collections::BTreeMap<String, Value> {
        self.values.lock().expect("memory lock").clone()
    }

    fn save_context(
        &self,
        inputs: std::collections::BTreeMap<String, Value>,
        outputs: std::collections::BTreeMap<String, Value>,
    ) {
        let mut values = self.values.lock().expect("memory lock");
        values.extend(inputs);
        values.extend(outputs);
    }

    fn clear(&self) {
        self.values.lock().expect("memory lock").clear();
    }
}

struct StaticChatLoader;

impl BaseChatLoader for StaticChatLoader {
    fn lazy_load<'a>(&'a self) -> Box<dyn Iterator<Item = ChatSession> + 'a> {
        Box::new(
            vec![
                ChatSession::new()
                    .with_messages(vec![HumanMessage::new("hello").into()])
                    .with_functions(vec![json!({"name": "lookup"})]),
            ]
            .into_iter(),
        )
    }
}

#[test]
fn classic_reexports_cache_and_globals() {
    set_debug(false);
    set_verbose(false);
    set_llm_cache(None);

    let cache: Arc<dyn langchain_classic::cache::BaseCache> = Arc::new(InMemoryCache::new());
    set_debug(true);
    set_verbose(true);
    set_llm_cache(Some(cache));

    assert!(get_debug());
    assert!(get_verbose());
    assert!(get_llm_cache().is_some());

    set_debug(false);
    set_verbose(false);
    set_llm_cache(None);
}

#[test]
fn classic_reexports_text_splitter_surface() {
    let character = CharacterTextSplitter::new(" ".to_owned(), 5, 0);
    let docs = character.create_documents(["alpha beta"], None);
    assert_eq!(docs.len(), 2);

    let recursive = RecursiveCharacterTextSplitter::new(5, 1, vec![" ".to_owned(), "".to_owned()]);
    let chunks = recursive.split_text("alpha beta gamma");
    assert!(!chunks.is_empty());

    let tokenizer = Tokenizer::whitespace();
    let token_chunks = split_text_on_tokens("alpha beta gamma", &tokenizer, 2, 1);
    assert_eq!(token_chunks, vec!["alpha beta", "beta gamma"]);

    let token_splitter = TokenTextSplitter::new(Tokenizer::whitespace(), 2, 0);
    let token_docs = token_splitter.create_documents(["alpha beta gamma"], None);
    assert_eq!(token_docs.len(), 2);
}

#[test]
fn classic_root_reexports_prompt_template_and_alias() {
    let prompt = PromptTemplate::new("Hello {name}");
    let alias = Prompt::new("Hi {name}");

    let rendered = prompt
        .format(
            &[(
                "name".to_owned(),
                PromptArgument::String("LangChain".to_owned()),
            )]
            .into(),
        )
        .expect("prompt template should render");
    let alias_rendered = alias
        .format(&[("name".to_owned(), PromptArgument::String("Rust".to_owned()))].into())
        .expect("prompt alias should render");

    assert_eq!(rendered, "Hello LangChain");
    assert_eq!(alias_rendered, "Hi Rust");
}

#[test]
fn classic_legacy_helpers_cover_base_language_env_formatting_and_input() {
    fn assert_base_language_model(_model: &BaseLanguageModel) {}

    let model = langchain_classic::chat_models::ParrotChatModel::new("classic", 8);
    assert_base_language_model(&model);

    let env = get_runtime_environment();
    assert_eq!(env.library, "langchain-classic");
    assert_eq!(env.runtime, "rust");

    let strict = StrictFormatter::new();
    let formatted = strict
        .format(
            "Hello {name}",
            &[("name".to_owned(), "LangChain".to_owned())].into(),
        )
        .expect("strict formatter should render named variables");
    let via_default = formatter()
        .format(
            "Hi {name}",
            &[("name".to_owned(), "Rust".to_owned())].into(),
        )
        .expect("default formatter should render named variables");
    assert_eq!(formatted, "Hello LangChain");
    assert_eq!(via_default, "Hi Rust");

    let mapping = get_color_mapping(vec!["a".to_owned(), "b".to_owned()], None)
        .expect("colors should be assigned");
    assert_eq!(mapping["a"], "blue");
    assert!(get_colored_text("hello", "blue").contains("hello"));
    assert!(get_bolded_text("hi").contains("hi"));
}

#[tokio::test]
async fn classic_base_memory_trait_supports_sync_and_async_paths() {
    let memory = RecorderMemory::default();
    memory.save_context(
        [("input".to_owned(), json!("hello"))].into(),
        [("output".to_owned(), json!("world"))].into(),
    );

    assert_eq!(
        memory.memory_variables(),
        vec!["input".to_owned(), "output".to_owned()]
    );
    assert_eq!(
        memory
            .load_memory_variables(Default::default())
            .get("output"),
        Some(&json!("world"))
    );
    assert_eq!(
        memory
            .aload_memory_variables(Default::default())
            .await
            .get("input"),
        Some(&json!("hello"))
    );
    memory.clear();
    assert!(memory.memory_variables().is_empty());
}

#[tokio::test]
async fn classic_reexports_callbacks_chat_loaders_load_storage_and_utils() {
    let run = CallbackRun::from_config(CallbackRunConfig::default().with_name("classic"));
    let event = CallbackEvent::custom("custom", json!({"ok": true}), Some(run.id()));
    assert_eq!(event.kind(), CallbackEventKind::Custom);
    assert_eq!(event.run_id(), run.id());

    let loader = StaticChatLoader;
    let sessions = loader.load();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].messages()[0].content(), "hello");
    assert_eq!(sessions[0].functions()[0]["name"], "lookup");

    let serialized = dumpd(&HumanMessage::new("hello")).expect("dumpd should serialize");
    match &serialized {
        SerializedValue::Constructor { id, .. } => {
            assert_eq!(id.last().map(String::as_str), Some("HumanMessage"));
        }
        SerializedValue::NotImplemented { .. } => {
            panic!("human message should serialize as constructor")
        }
    }
    let revived = Reviver::core()
        .revive(serialized)
        .expect("reviver should reconstruct human messages");
    assert_eq!(
        revived.lc_id().last().map(String::as_str),
        Some("HumanMessage")
    );

    let store = InMemoryStore::<Value>::new();
    store.mset(vec![("alpha".to_owned(), json!("A"))]);
    assert_eq!(store.mget(&["alpha".to_owned()])[0], Some(json!("A")));
    store.amset(vec![("beta".to_owned(), json!("B"))]).await;
    assert_eq!(
        store.amget(vec!["beta".to_owned()]).await[0],
        Some(json!("B"))
    );

    let formatted = langchain_classic::utils::formatting::formatter()
        .format(
            "Hello {name}",
            &[("name".to_owned(), "Rust".to_owned())].into(),
        )
        .expect("nested utils formatting facade should render");
    assert_eq!(formatted, "Hello Rust");
    assert!(langchain_classic::utils::input::get_bolded_text("hi").contains("hi"));
}

#[test]
fn classic_utils_schema_docstore_and_indexes_match_reference_facades() {
    let rendered = stringify_dict(&serde_json::Map::from_iter([(
        "status".to_owned(),
        json!("ok"),
    )]));
    assert!(rendered.contains("status: ok"));
    assert_eq!(comma_list(["alpha", "beta", "gamma"]), "alpha, beta, gamma");

    unsafe {
        std::env::set_var("LANGCHAIN_CLASSIC_TEST_ENV", "available");
    }
    assert_eq!(
        get_from_env("api_key", "LANGCHAIN_CLASSIC_TEST_ENV", None)
            .expect("env helper should resolve classic utility values"),
        "available"
    );
    unsafe {
        std::env::remove_var("LANGCHAIN_CLASSIC_TEST_ENV");
    }

    let docstore = langchain_classic::docstore::InMemoryDocstore::new()
        .with_document("doc-1", Document::new("classic docstore"));
    assert_eq!(
        docstore.search("doc-1"),
        Some(Document::new("classic docstore"))
    );

    let callback_error = langchain_classic::schema::LangChainException::request("boom");
    assert!(callback_error.to_string().contains("boom"));
    assert_eq!(langchain_classic::schema::RUN_KEY, "__run");

    let doc = langchain_classic::indexes::Document::new("indexed alias");
    assert_eq!(doc.page_content, "indexed alias");
}

#[tokio::test]
async fn classic_reexports_document_loaders_and_indexing() {
    let loader =
        langchain_classic::document_loaders::StaticDocumentLoader::new(vec![Document::new(
            "classic document loader",
        )]);
    let loaded = loader.load().await.expect("document loader should work");
    assert_eq!(loaded[0].page_content, "classic document loader");

    let mut vector_store = InMemoryVectorStore::new(CharacterEmbeddings::new());
    let mut record_manager = InMemoryRecordManager::new("classic");
    let result = index(
        vec![Document::new("classic indexing")],
        &mut vector_store,
        &mut record_manager,
        false,
    )
    .await
    .expect("classic indexing should work");
    assert_eq!(result.num_added, 1);

    let docs = vector_store
        .similarity_search("classic", 1)
        .await
        .expect("similarity search should work");
    assert_eq!(docs.len(), 1);
}
