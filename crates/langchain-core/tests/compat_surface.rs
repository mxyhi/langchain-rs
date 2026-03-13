use std::sync::Arc;

use futures_util::future::BoxFuture;
use langchain_core::caches::{BaseCache, InMemoryCache};
use langchain_core::chat_history::InMemoryChatMessageHistory;
use langchain_core::chat_sessions::ChatSession;
use langchain_core::cross_encoders::{BaseCrossEncoder, TextPair};
use langchain_core::exceptions::LangChainException;
use langchain_core::globals::{
    get_debug, get_llm_cache, get_verbose, set_debug, set_llm_cache, set_verbose,
};
use langchain_core::messages::{AIMessage, HumanMessage};
use langchain_core::outputs::Generation;
use langchain_core::version::VERSION as VERSION_MODULE;

struct StaticCrossEncoder;

impl BaseCrossEncoder for StaticCrossEncoder {
    fn score<'a>(
        &'a self,
        text_pairs: Vec<TextPair>,
    ) -> BoxFuture<'a, Result<Vec<f32>, langchain_core::LangChainError>> {
        Box::pin(async move {
            Ok(text_pairs
                .into_iter()
                .map(|(left, right)| if left == right { 1.0 } else { 0.25 })
                .collect())
        })
    }
}

#[test]
fn in_memory_cache_round_trips_generations() {
    let cache = InMemoryCache::new();

    assert!(cache.lookup("prompt", "llm").is_none());

    cache.update("prompt", "llm", vec![Generation::new("Paris").into()]);

    let cached = cache
        .lookup("prompt", "llm")
        .expect("cache should return stored generations");
    assert_eq!(cached[0].text(), "Paris");

    cache.clear();
    assert!(cache.lookup("prompt", "llm").is_none());
}

#[test]
fn globals_round_trip_debug_verbose_and_llm_cache() {
    set_debug(false);
    set_verbose(false);
    set_llm_cache(None);

    let cache: Arc<dyn BaseCache> = Arc::new(InMemoryCache::new());
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

#[tokio::test]
async fn in_memory_chat_history_tracks_messages() {
    let history = InMemoryChatMessageHistory::new();

    history.add_user_message("hello");
    history.add_ai_message("world");

    let messages = history.aget_messages().await;
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].content(), "hello");
    assert_eq!(messages[1].content(), "world");

    history.clear();
    assert!(history.messages().is_empty());
}

#[test]
fn chat_session_keeps_messages_and_functions() {
    let session = ChatSession::new()
        .with_messages(vec![HumanMessage::new("hello").into()])
        .with_functions(vec![serde_json::json!({"name": "lookup"})]);

    assert_eq!(session.messages().len(), 1);
    assert_eq!(session.messages()[0].content(), "hello");
    assert_eq!(session.functions().len(), 1);
    assert_eq!(session.functions()[0]["name"], "lookup");
}

#[test]
fn chat_history_accepts_message_instances() {
    let history = InMemoryChatMessageHistory::new();

    history.add_message(HumanMessage::new("hi").into());
    history.add_message(AIMessage::new("there").into());

    assert_eq!(history.messages().len(), 2);
    assert_eq!(history.messages()[0].content(), "hi");
    assert_eq!(history.messages()[1].content(), "there");
}

#[test]
fn exceptions_and_version_are_addressable_through_python_like_paths() {
    let error = LangChainException::request("boom");

    assert!(error.to_string().contains("boom"));
    assert_eq!(langchain_core::VERSION, VERSION_MODULE);
}

#[tokio::test]
async fn cross_encoder_trait_scores_text_pairs() {
    let encoder = StaticCrossEncoder;
    let scores = encoder
        .score(vec![
            ("alpha".to_owned(), "alpha".to_owned()),
            ("alpha".to_owned(), "beta".to_owned()),
        ])
        .await
        .expect("cross encoder scoring should succeed");

    assert_eq!(scores, vec![1.0, 0.25]);
}
