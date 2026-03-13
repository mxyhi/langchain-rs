use langchain_core::documents::Document;
use langchain_core::indexing::{IndexingResult, RecordManager, index};
use langchain_core::vectorstores::VectorStore;

pub async fn assert_indexing_result<V, R>(
    vector_store: &mut V,
    record_manager: &mut R,
    documents: Vec<Document>,
    cleanup: bool,
    expected: IndexingResult,
) -> IndexingResult
where
    V: VectorStore + Send + Sync,
    R: RecordManager,
{
    let result = index(documents, vector_store, record_manager, cleanup)
        .await
        .expect("indexing should succeed");
    assert_eq!(result, expected);
    result
}
