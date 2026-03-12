use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::runnables::{Runnable, RunnableConfig};
use serde::{Deserialize, Serialize};

use crate::types::WebSearchOptions;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerplexitySearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub score: f32,
}

impl PerplexitySearchHit {
    pub fn new(
        title: impl Into<String>,
        url: impl Into<String>,
        snippet: impl Into<String>,
        score: f32,
    ) -> Self {
        Self {
            title: title.into(),
            url: url.into(),
            snippet: snippet.into(),
            score,
        }
    }
}

fn score(query: &str, snippet: &str) -> f32 {
    let normalized_query = query.to_ascii_lowercase();
    let normalized_snippet = snippet.to_ascii_lowercase();
    normalized_query
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .map(|token| normalized_snippet.matches(token).count() as f32)
        .sum()
}

#[derive(Debug, Clone, Default)]
pub struct PerplexitySearchResults {
    corpus: Vec<PerplexitySearchHit>,
    max_results: usize,
    search_options: WebSearchOptions,
}

impl PerplexitySearchResults {
    pub fn new() -> Self {
        Self {
            corpus: Vec::new(),
            max_results: 5,
            search_options: WebSearchOptions::default(),
        }
    }

    pub fn with_hit(mut self, hit: PerplexitySearchHit) -> Self {
        self.corpus.push(hit);
        self
    }

    pub fn with_hits(mut self, hits: impl IntoIterator<Item = PerplexitySearchHit>) -> Self {
        self.corpus.extend(hits);
        self
    }

    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results.max(1);
        self
    }

    pub fn with_search_options(mut self, search_options: WebSearchOptions) -> Self {
        self.search_options = search_options;
        self
    }

    pub fn search(&self, query: &str) -> Vec<PerplexitySearchHit> {
        let context_bonus = self.search_options.search_context_size.unwrap_or(1) as f32;
        let mut ranked = self
            .corpus
            .iter()
            .map(|hit| {
                let mut ranked_hit = hit.clone();
                ranked_hit.score = score(query, &ranked_hit.snippet) + context_bonus;
                ranked_hit
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
        ranked.truncate(self.max_results);
        ranked
    }
}

impl Runnable<String, Vec<PerplexitySearchHit>> for PerplexitySearchResults {
    fn invoke<'a>(
        &'a self,
        input: String,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<Vec<PerplexitySearchHit>, LangChainError>> {
        Box::pin(async move { Ok(self.search(&input)) })
    }
}
