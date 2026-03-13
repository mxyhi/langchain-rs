use futures_util::future::BoxFuture;

use crate::LangChainError;

/// Canonical text-pair input for cross-encoder scoring.
pub type TextPair = (String, String);

/// Interface for cross-encoder models that score text-pair similarity.
///
/// The reference Python surface exposes a `BaseCrossEncoder.score(...)` contract.
/// The Rust port keeps the same conceptual boundary while making scoring async so
/// remote and local implementations share one trait.
pub trait BaseCrossEncoder: Send + Sync {
    fn score<'a>(
        &'a self,
        text_pairs: Vec<TextPair>,
    ) -> BoxFuture<'a, Result<Vec<f32>, LangChainError>>;

    fn score_pair<'a>(
        &'a self,
        left: impl Into<String>,
        right: impl Into<String>,
    ) -> BoxFuture<'a, Result<f32, LangChainError>>
    where
        Self: Sized,
    {
        let text_pairs = vec![(left.into(), right.into())];
        Box::pin(async move {
            self.score(text_pairs)
                .await?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    LangChainError::request("cross encoder returned no score for a single pair")
                })
        })
    }
}
