use langchain_core::documents::Document;

pub trait TextSplitter {
    fn split_text(&self, text: &str) -> Vec<String>;

    fn create_documents<I, S>(
        &self,
        texts: I,
        _metadatas: Option<Vec<serde_json::Value>>,
    ) -> Vec<Document>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        texts
            .into_iter()
            .flat_map(|text| self.split_text(text.as_ref()))
            .map(Document::new)
            .collect()
    }
}
