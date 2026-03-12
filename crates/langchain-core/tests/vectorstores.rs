use langchain_core::documents::Document;
use langchain_core::embeddings::CharacterEmbeddings;
use langchain_core::vectorstores::{InMemoryVectorStore, VectorStore, metadata};
use serde_json::json;

struct WithoutGetByIdsVectorStore(InMemoryVectorStore<CharacterEmbeddings>);

impl WithoutGetByIdsVectorStore {
    fn new() -> Self {
        Self(InMemoryVectorStore::new(CharacterEmbeddings::new()))
    }
}

impl VectorStore for WithoutGetByIdsVectorStore {
    fn add_documents<'a>(
        &'a mut self,
        documents: Vec<Document>,
    ) -> futures_util::future::BoxFuture<'a, Result<Vec<String>, langchain_core::LangChainError>>
    {
        self.0.add_documents(documents)
    }

    fn similarity_search<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> futures_util::future::BoxFuture<'a, Result<Vec<Document>, langchain_core::LangChainError>>
    {
        self.0.similarity_search(query, limit)
    }
}

#[tokio::test]
async fn in_memory_vectorstore_similarity_search_returns_best_match() {
    let mut vectorstore = InMemoryVectorStore::new(CharacterEmbeddings::new());
    let mut docs = vec![
        Document::new("apple pie recipe"),
        Document::new("ocean weather report"),
        Document::new("apple orchard guide"),
    ];
    docs[0].metadata = metadata("kind", json!("recipe"));

    let ids = vectorstore
        .add_documents(docs)
        .await
        .expect("documents should be stored");

    assert_eq!(ids.len(), 3);

    let results = vectorstore
        .similarity_search("apple dessert", 2)
        .await
        .expect("search should succeed");

    assert_eq!(results.len(), 2);
    assert!(results[0].page_content.contains("apple"));
}

#[tokio::test]
async fn in_memory_vectorstore_get_by_ids_returns_matching_documents_only() {
    let mut vectorstore = InMemoryVectorStore::new(CharacterEmbeddings::new());
    let ids = vectorstore
        .add_documents(vec![Document::new("alpha"), Document::new("beta")])
        .await
        .expect("documents should be stored");

    let retrieved = vectorstore
        .get_by_ids(&[ids[1].clone(), "missing".to_owned()])
        .expect("get_by_ids should not fail");

    assert_eq!(retrieved.len(), 1);
    assert_eq!(retrieved[0].id.as_deref(), Some(ids[1].as_str()));
    assert_eq!(retrieved[0].page_content, "beta");
}

#[test]
fn vectorstore_without_get_by_ids_keeps_standard_failure_message() {
    let vectorstore = WithoutGetByIdsVectorStore::new();
    let error = vectorstore
        .get_by_ids(&["id1".to_owned(), "id2".to_owned()])
        .expect_err("missing get_by_ids implementation should fail");

    assert_eq!(
        error.to_string(),
        "unsupported operation: WithoutGetByIdsVectorStore does not yet support get_by_ids"
    );
}
