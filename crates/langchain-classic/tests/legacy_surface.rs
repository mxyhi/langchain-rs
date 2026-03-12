use std::sync::Arc;

use langchain_classic::{Prompt, PromptTemplate};
use langchain_classic::cache::InMemoryCache;
use langchain_classic::globals::{
    get_debug, get_llm_cache, get_verbose, set_debug, set_llm_cache, set_verbose,
};
use langchain_classic::prompts::PromptArgument;
use langchain_classic::text_splitter::{
    CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter, TokenTextSplitter,
    Tokenizer, split_text_on_tokens,
};

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
