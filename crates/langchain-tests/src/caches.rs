use langchain_core::caches::{BaseCache, CacheValue};
use langchain_core::outputs::{Generation, GenerationCandidate};

pub fn cache_value(text: &str) -> CacheValue {
    vec![GenerationCandidate::from(Generation::new(text))]
}

pub fn assert_cache_lookup<C>(cache: &C, prompt: &str, llm_string: &str, expected_text: &str)
where
    C: BaseCache,
{
    let hit = cache
        .lookup(prompt, llm_string)
        .expect("cache entry should exist");
    assert_eq!(hit.first().expect("cache entry").text(), expected_text);
}

pub async fn assert_async_cache_lookup<C>(
    cache: &C,
    prompt: &str,
    llm_string: &str,
    expected_text: &str,
) where
    C: BaseCache,
{
    let hit = cache
        .alookup(prompt, llm_string)
        .await
        .expect("cache entry should exist");
    assert_eq!(hit.first().expect("cache entry").text(), expected_text);
}

pub async fn assert_cache_round_trip<C>(
    cache: &C,
    prompt: &str,
    llm_string: &str,
    value: CacheValue,
) where
    C: BaseCache,
{
    cache.update(prompt, llm_string, value.clone());
    let expected = value.first().expect("cache entry").text().to_owned();
    assert_cache_lookup(cache, prompt, llm_string, &expected);

    cache.aupdate(prompt, llm_string, value).await;
    assert_async_cache_lookup(cache, prompt, llm_string, &expected).await;
}

pub async fn assert_cache_clear<C>(cache: &C, prompt: &str, llm_string: &str, value: CacheValue)
where
    C: BaseCache,
{
    cache.update(prompt, llm_string, value);
    cache.clear();
    assert!(cache.lookup(prompt, llm_string).is_none());

    cache
        .aupdate(prompt, llm_string, cache_value("reloaded"))
        .await;
    cache.aclear().await;
    assert!(cache.alookup(prompt, llm_string).await.is_none());
}
