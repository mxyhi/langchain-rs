use std::collections::HashMap;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use langchain_core::document_loaders::{BaseLoader, TextLoader};
use langchain_core::embeddings::CharacterEmbeddings;
use langchain_core::example_selectors::{
    BaseExampleSelector, LengthBasedExampleSelector, SemanticSimilarityExampleSelector,
};
use langchain_core::indexing::{InMemoryRecordManager, RecordManager};
use langchain_core::prompts::{PromptArgument, PromptArguments};
use serde_json::Value;

fn prompt_arguments(question: &str) -> PromptArguments {
    HashMap::from([(
        "question".to_owned(),
        PromptArgument::String(question.to_owned()),
    )])
}

#[tokio::test]
async fn text_loader_reads_utf8_file_and_tracks_source() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("langchain-rs-text-loader-{unique}.txt"));
    fs::write(&path, "alpha\nbeta").expect("temp file should be writable");

    let loader = TextLoader::new(&path);
    let documents = loader.load().await.expect("text loader should read utf8");

    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].page_content, "alpha\nbeta");
    assert_eq!(
        documents[0].metadata.get("source"),
        Some(&Value::String(path.display().to_string()))
    );

    let _ = fs::remove_file(path);
}

#[tokio::test]
async fn length_based_selector_trims_examples_as_input_grows() {
    let selector = LengthBasedExampleSelector::new(150).with_examples(vec![
        prompt_arguments("Question: what is Rust?\nAnswer: a systems language"),
        prompt_arguments("Question: what is Tokio?\nAnswer: an async runtime"),
        prompt_arguments("Question: what is Qdrant?\nAnswer: a vector database"),
    ]);

    let short = selector
        .select_examples(&prompt_arguments("Short question?"))
        .await
        .expect("selector should succeed");
    assert!(short.len() >= 2);

    let long_question =
        "This question is intentionally long enough to force the selector to trim examples. "
            .repeat(8);
    let long = selector
        .select_examples(&prompt_arguments(&long_question))
        .await
        .expect("selector should still succeed");
    assert!(long.len() < short.len());
}

#[tokio::test]
async fn semantic_similarity_selector_prefers_closest_example() {
    let mut selector = SemanticSimilarityExampleSelector::new(CharacterEmbeddings::new(), 1);
    selector
        .add_example(prompt_arguments("cats like warm boxes"))
        .await
        .expect("first example should be indexable");
    selector
        .add_example(prompt_arguments("cars need fuel"))
        .await
        .expect("second example should be indexable");

    let selected = selector
        .select_examples(&prompt_arguments("cat nap"))
        .await
        .expect("selector should return the closest example");

    assert_eq!(selected.len(), 1);
    assert_eq!(
        selected[0].get("question"),
        Some(&PromptArgument::String("cats like warm boxes".to_owned()))
    );
}

#[tokio::test]
async fn in_memory_record_manager_tracks_groups_and_deletes_keys() {
    let mut manager = InMemoryRecordManager::new("docs");
    manager
        .acreate_schema()
        .await
        .expect("schema creation should be a no-op");
    manager
        .aupdate(vec!["doc-1".to_owned(), "doc-2".to_owned()], None)
        .await
        .expect("ungrouped keys should insert");
    manager
        .aupdate(vec!["doc-3".to_owned()], Some(vec!["batch-a".to_owned()]))
        .await
        .expect("grouped keys should insert");

    assert_eq!(
        manager
            .aexists(vec!["doc-1".to_owned(), "missing".to_owned()])
            .await
            .expect("exists should succeed"),
        vec![true, false]
    );
    assert_eq!(
        manager
            .alist_keys(None, None, Some(vec!["batch-a".to_owned()]), None)
            .await
            .expect("group filtering should succeed"),
        vec!["doc-3".to_owned()]
    );

    manager
        .adelete_keys(vec!["doc-2".to_owned()])
        .await
        .expect("delete should succeed");
    assert_eq!(
        manager
            .alist_keys(None, None, None, None)
            .await
            .expect("listing should succeed"),
        vec!["doc-1".to_owned(), "doc-3".to_owned()]
    );
}
