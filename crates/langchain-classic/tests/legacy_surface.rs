use std::sync::Arc;

use langchain_classic::base_language::BaseLanguageModel;
use langchain_classic::base_memory::BaseMemory;
use langchain_classic::cache::InMemoryCache;
use langchain_classic::env::get_runtime_environment;
use langchain_classic::formatting::{StrictFormatter, formatter};
use langchain_classic::globals::{
    get_debug, get_llm_cache, get_verbose, set_debug, set_llm_cache, set_verbose,
};
use langchain_classic::input::{get_bolded_text, get_color_mapping, get_colored_text};
use langchain_classic::prompts::PromptArgument;
use langchain_classic::text_splitter::{
    CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter, TokenTextSplitter,
    Tokenizer, split_text_on_tokens,
};
use langchain_classic::{Prompt, PromptTemplate};
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
