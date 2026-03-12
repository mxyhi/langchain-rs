use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::documents::Document;
use langchain_core::runnables::{Runnable, RunnableConfig};
use serde_json::json;

use crate::tools::{ExaSearchResults, SearchHit};

#[derive(Debug, Clone, Default)]
pub struct ExaSearchRetriever {
    tool: ExaSearchResults,
}

impl ExaSearchRetriever {
    pub fn new() -> Self {
        Self {
            tool: ExaSearchResults::new(),
        }
    }

    pub fn with_hit(mut self, hit: SearchHit) -> Self {
        self.tool = self.tool.with_hit(hit);
        self
    }

    pub fn with_hits(mut self, hits: impl IntoIterator<Item = SearchHit>) -> Self {
        self.tool = self.tool.with_hits(hits);
        self
    }

    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.tool = self.tool.with_max_results(max_results);
        self
    }
}

impl Runnable<String, Vec<Document>> for ExaSearchRetriever {
    fn invoke<'a>(
        &'a self,
        input: String,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<Vec<Document>, LangChainError>> {
        Box::pin(async move {
            let hits = self.tool.invoke(input, config).await?;
            Ok(hits
                .into_iter()
                .map(|hit| {
                    let mut document = Document::new(hit.text);
                    document
                        .metadata
                        .insert("title".to_owned(), json!(hit.title));
                    document.metadata.insert("url".to_owned(), json!(hit.url));
                    document
                        .metadata
                        .insert("score".to_owned(), json!(hit.score));
                    document
                })
                .collect())
        })
    }
}
