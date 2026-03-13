use langchain_core::documents::Document;
use langchain_core::indexing::DocumentIndex;
use langchain_core::retrievers::BaseRetriever;

pub trait DocumentIndexHarness {
    type Index: DocumentIndex;

    fn index(&self) -> Self::Index;

    fn seed_documents(&self) -> Vec<Document>;

    fn overwrite_document(&self) -> Document;

    fn query(&self) -> &'static str;

    fn expected_page_content(&self) -> &'static str;
}

pub struct DocumentIndexIntegrationTests<H> {
    harness: H,
}

impl<H> DocumentIndexIntegrationTests<H> {
    pub fn new(harness: H) -> Self {
        Self { harness }
    }
}

impl<H> DocumentIndexIntegrationTests<H>
where
    H: DocumentIndexHarness,
{
    pub async fn run(&self) {
        let mut index = self.harness.index();
        let seed_documents = self.harness.seed_documents();

        let upsert = index
            .upsert_documents(seed_documents)
            .await
            .expect("document index should accept seed documents");
        assert_eq!(upsert.num_skipped, 0);
        assert!(upsert.num_added >= 1);

        let initial = index
            .get_relevant_documents(self.harness.query(), Default::default())
            .await
            .expect("document index should return relevant documents");
        assert!(
            initial
                .iter()
                .any(|document| document.page_content == self.harness.expected_page_content())
        );

        let overwrite = self.harness.overwrite_document();
        index
            .upsert_documents(vec![overwrite.clone()])
            .await
            .expect("document index should overwrite matching ids");
        let updated = index
            .get_relevant_documents(self.harness.query(), Default::default())
            .await
            .expect("document index should still retrieve updated documents");
        assert!(
            updated
                .iter()
                .any(|document| document.page_content == overwrite.page_content)
        );

        let deleted = index
            .delete_documents(vec!["doc-2".to_owned(), "missing".to_owned()])
            .await
            .expect("document index delete should succeed");
        assert_eq!(deleted.num_deleted, 1);
    }
}
