use std::collections::BTreeMap;

use langchain_core::document_loaders::{
    BaseBlobParser, BaseLoader, Blob, BlobLoader, LangSmithLoader, StaticBlobLoader,
    StaticDocumentLoader,
};
use langchain_core::documents::Document;
use langchain_core::embeddings::CharacterEmbeddings;
use langchain_core::example_selectors::{
    BaseExampleSelector, LengthBasedExampleSelector, sorted_values,
};
use langchain_core::indexing::{InMemoryRecordManager, index};
use langchain_core::prompts::{PromptArgument, PromptArguments};
use langchain_core::vectorstores::{InMemoryVectorStore, VectorStore};
use serde_json::json;

struct Utf8BlobParser;

impl BaseBlobParser for Utf8BlobParser {
    fn parse<'a>(
        &'a self,
        blob: &'a Blob,
    ) -> futures_util::future::BoxFuture<'a, Result<Vec<Document>, langchain_core::LangChainError>>
    {
        Box::pin(async move {
            Ok(vec![Document::new(
                String::from_utf8(blob.as_bytes().to_vec()).expect("utf8"),
            )])
        })
    }
}

#[tokio::test]
async fn document_loaders_cover_static_blob_and_langsmith_boundaries() {
    let documents = vec![Document::new("alpha"), Document::new("beta")];
    let loader = StaticDocumentLoader::new(documents.clone());
    assert_eq!(
        loader.load().await.expect("static loader should work"),
        documents
    );

    let blob = Blob::from_bytes("hello");
    let blob_loader = StaticBlobLoader::new(vec![blob.clone()]);
    assert_eq!(
        blob_loader
            .load_blobs()
            .await
            .expect("blob loader should work"),
        vec![blob.clone()]
    );

    let parser = Utf8BlobParser;
    let parsed = parser.parse(&blob).await.expect("blob parser should work");
    assert_eq!(parsed[0].page_content, "hello");

    let langsmith_loader = LangSmithLoader::new(vec![Document::new("trace doc")]);
    assert_eq!(
        langsmith_loader
            .load()
            .await
            .expect("langsmith boundary should work")[0]
            .page_content,
        "trace doc"
    );
}

#[tokio::test]
async fn example_selectors_cover_length_and_sorted_values_helpers() {
    let mut selector = LengthBasedExampleSelector::new(24);
    selector.add_example(PromptArguments::from([(
        "input".to_owned(),
        PromptArgument::String("short".to_owned()),
    )]));
    selector.add_example(PromptArguments::from([(
        "input".to_owned(),
        PromptArgument::String("this is too long to fit".to_owned()),
    )]));

    let selected = selector
        .select_examples(&PromptArguments::from([(
            "question".to_owned(),
            PromptArgument::String("ping".to_owned()),
        )]))
        .await
        .expect("selection should succeed");
    assert_eq!(selected.len(), 1);
    assert_eq!(sorted_values(&selected[0]), vec!["short".to_owned()]);
}

#[tokio::test]
async fn indexing_adds_skips_and_cleans_up_documents() {
    let mut store = InMemoryVectorStore::new(CharacterEmbeddings::new());
    let mut records = InMemoryRecordManager::new("tests");

    let first = index(
        vec![
            Document::new("alpha"),
            Document {
                page_content: "beta".to_owned(),
                metadata: BTreeMap::from([("topic".to_owned(), json!("letters"))]),
                id: None,
            },
        ],
        &mut store,
        &mut records,
        false,
    )
    .await
    .expect("first index should succeed");
    assert_eq!(first.num_added, 2);
    assert_eq!(first.num_skipped, 0);

    let second = index(
        vec![
            Document::new("alpha"),
            Document {
                page_content: "beta".to_owned(),
                metadata: BTreeMap::from([("topic".to_owned(), json!("letters"))]),
                id: None,
            },
        ],
        &mut store,
        &mut records,
        false,
    )
    .await
    .expect("second index should succeed");
    assert_eq!(second.num_added, 0);
    assert_eq!(second.num_skipped, 2);

    let cleanup = index(vec![Document::new("alpha")], &mut store, &mut records, true)
        .await
        .expect("cleanup should succeed");
    assert_eq!(cleanup.num_deleted, 1);

    let docs = store
        .similarity_search("alpha", 10)
        .await
        .expect("search should work");
    assert_eq!(docs.len(), 1);
}
