use langchain_core::caches::{BaseCache, CacheValue};
use langchain_core::outputs::{Generation, GenerationCandidate};

pub trait SyncCacheHarness {
    type Cache: BaseCache;

    fn cache(&self) -> Self::Cache;

    fn sample_prompt(&self) -> &'static str {
        "Sample prompt for testing."
    }

    fn sample_llm_string(&self) -> &'static str {
        "Sample LLM string configuration."
    }

    fn sample_generation(&self) -> CacheValue {
        vec![GenerationCandidate::from(Generation::new(
            "Sample generated text.",
        ))]
    }
}

pub struct SyncCacheTestSuite<H> {
    harness: H,
}

impl<H> SyncCacheTestSuite<H> {
    pub fn new(harness: H) -> Self {
        Self { harness }
    }
}

impl<H> SyncCacheTestSuite<H>
where
    H: SyncCacheHarness,
{
    pub fn run(&self) {
        let cache = self.harness.cache();
        assert!(
            cache
                .lookup(
                    self.harness.sample_prompt(),
                    self.harness.sample_llm_string()
                )
                .is_none()
        );

        cache.update(
            self.harness.sample_prompt(),
            self.harness.sample_llm_string(),
            self.harness.sample_generation(),
        );
        let hit = cache
            .lookup(
                self.harness.sample_prompt(),
                self.harness.sample_llm_string(),
            )
            .expect("cache hit should exist");
        assert_eq!(hit.len(), 1);

        cache.clear();
        assert!(
            cache
                .lookup(
                    self.harness.sample_prompt(),
                    self.harness.sample_llm_string()
                )
                .is_none()
        );
    }
}

pub trait AsyncCacheHarness: SyncCacheHarness {}

pub struct AsyncCacheTestSuite<H> {
    harness: H,
}

impl<H> AsyncCacheTestSuite<H> {
    pub fn new(harness: H) -> Self {
        Self { harness }
    }
}

impl<H> AsyncCacheTestSuite<H>
where
    H: AsyncCacheHarness,
{
    pub async fn run(&self) {
        let cache = self.harness.cache();
        assert!(
            cache
                .alookup(
                    self.harness.sample_prompt(),
                    self.harness.sample_llm_string()
                )
                .await
                .is_none()
        );

        cache
            .aupdate(
                self.harness.sample_prompt(),
                self.harness.sample_llm_string(),
                self.harness.sample_generation(),
            )
            .await;

        let hit = cache
            .alookup(
                self.harness.sample_prompt(),
                self.harness.sample_llm_string(),
            )
            .await
            .expect("async cache hit should exist");
        assert_eq!(hit.len(), 1);

        cache.aclear().await;
        assert!(
            cache
                .alookup(
                    self.harness.sample_prompt(),
                    self.harness.sample_llm_string()
                )
                .await
                .is_none()
        );
    }
}
