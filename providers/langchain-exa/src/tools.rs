use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::runnables::{Runnable, RunnableConfig};
use serde::{Deserialize, Serialize};

use crate::types::TextContentsOptions;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub text: String,
    pub score: f32,
}

impl SearchHit {
    pub fn new(
        title: impl Into<String>,
        url: impl Into<String>,
        text: impl Into<String>,
        score: f32,
    ) -> Self {
        Self {
            title: title.into(),
            url: url.into(),
            text: text.into(),
            score,
        }
    }
}

fn score(query: &str, candidate: &str) -> f32 {
    let normalized_query = query.to_ascii_lowercase();
    let normalized_candidate = candidate.to_ascii_lowercase();

    normalized_query
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .map(|token| normalized_candidate.matches(token).count() as f32)
        .sum()
}

fn truncate_text(text: &str, options: &TextContentsOptions) -> String {
    match options.max_characters {
        Some(limit) => text.chars().take(limit).collect(),
        None => text.to_owned(),
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExaSearchResults {
    corpus: Vec<SearchHit>,
    max_results: usize,
    text_options: TextContentsOptions,
}

impl ExaSearchResults {
    pub fn new() -> Self {
        Self {
            corpus: Vec::new(),
            max_results: 5,
            text_options: TextContentsOptions::default(),
        }
    }

    pub fn with_hit(mut self, hit: SearchHit) -> Self {
        self.corpus.push(hit);
        self
    }

    pub fn with_hits(mut self, hits: impl IntoIterator<Item = SearchHit>) -> Self {
        self.corpus.extend(hits);
        self
    }

    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results.max(1);
        self
    }

    pub fn with_text_options(mut self, text_options: TextContentsOptions) -> Self {
        self.text_options = text_options;
        self
    }

    pub fn search(&self, query: &str) -> Vec<SearchHit> {
        let mut ranked = self
            .corpus
            .iter()
            .map(|hit| {
                let mut ranked_hit = hit.clone();
                ranked_hit.text = truncate_text(&ranked_hit.text, &self.text_options);
                ranked_hit.score =
                    score(query, &format!("{} {}", ranked_hit.title, ranked_hit.text));
                ranked_hit
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
        ranked.truncate(self.max_results);
        ranked
    }
}

impl Runnable<String, Vec<SearchHit>> for ExaSearchResults {
    fn invoke<'a>(
        &'a self,
        input: String,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<Vec<SearchHit>, LangChainError>> {
        Box::pin(async move { Ok(self.search(&input)) })
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExaFindSimilarResults {
    inner: ExaSearchResults,
}

impl ExaFindSimilarResults {
    pub fn new() -> Self {
        Self {
            inner: ExaSearchResults::new(),
        }
    }

    pub fn with_hit(mut self, hit: SearchHit) -> Self {
        self.inner = self.inner.with_hit(hit);
        self
    }

    pub fn with_hits(mut self, hits: impl IntoIterator<Item = SearchHit>) -> Self {
        self.inner = self.inner.with_hits(hits);
        self
    }

    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.inner = self.inner.with_max_results(max_results);
        self
    }

    pub fn find_similar(&self, seed_text: &str) -> Vec<SearchHit> {
        self.inner.search(seed_text)
    }
}

impl Runnable<String, Vec<SearchHit>> for ExaFindSimilarResults {
    fn invoke<'a>(
        &'a self,
        input: String,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<Vec<SearchHit>, LangChainError>> {
        self.inner.invoke(input, config)
    }
}
