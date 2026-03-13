use std::fmt::Debug;

use futures_util::StreamExt;
use langchain_core::stores::BaseStore;

fn assert_yielded_keys<S, V>(store: &S, expected_keys: &[String], prefix: Option<&str>)
where
    S: BaseStore<V>,
    V: Clone + Debug + PartialEq + Send + Sync + 'static,
{
    let yielded = store.yield_keys(prefix).collect::<Vec<_>>();
    for key in expected_keys
        .iter()
        .filter(|key| prefix.is_none_or(|prefix| key.starts_with(prefix)))
    {
        assert!(
            yielded.contains(key),
            "expected yielded keys to contain `{key}`"
        );
    }
}

async fn assert_async_yielded_keys<S, V>(store: &S, expected_keys: &[String], prefix: Option<&str>)
where
    S: BaseStore<V>,
    V: Clone + Debug + PartialEq + Send + Sync + 'static,
{
    let yielded = store.ayield_keys(prefix).collect::<Vec<_>>().await;
    for key in expected_keys
        .iter()
        .filter(|key| prefix.is_none_or(|prefix| key.starts_with(prefix)))
    {
        assert!(
            yielded.contains(key),
            "expected async yielded keys to contain `{key}`"
        );
    }
}

pub async fn assert_store_roundtrip<S, V>(
    store: &S,
    key_value_pairs: Vec<(String, V)>,
    prefix: Option<&str>,
) where
    S: BaseStore<V>,
    V: Clone + Debug + PartialEq + Send + Sync + 'static,
{
    let keys = key_value_pairs
        .iter()
        .map(|(key, _)| key.clone())
        .collect::<Vec<_>>();
    let expected_values = key_value_pairs
        .iter()
        .map(|(_, value)| Some(value.clone()))
        .collect::<Vec<_>>();

    store.mset(key_value_pairs.clone());
    assert_eq!(store.mget(&keys), expected_values);
    assert_yielded_keys(store, &keys, prefix);

    store.mdelete(&keys);
    assert!(store.mget(&keys).iter().all(Option::is_none));

    store.amset(key_value_pairs).await;
    assert_eq!(store.amget(keys.clone()).await, expected_values);
    assert_async_yielded_keys(store, &keys, prefix).await;

    store.amdelete(keys.clone()).await;
    assert!(store.amget(keys).await.iter().all(Option::is_none));
}

pub async fn assert_store_round_trip<S, V>(
    store: &S,
    key_value_pairs: Vec<(String, V)>,
    expected_keys: Vec<String>,
    prefix: Option<&str>,
) where
    S: BaseStore<V>,
    V: Clone + Debug + PartialEq + Send + Sync + 'static,
{
    let keys = key_value_pairs
        .iter()
        .map(|(key, _)| key.clone())
        .collect::<Vec<_>>();
    let expected_values = key_value_pairs
        .iter()
        .map(|(_, value)| Some(value.clone()))
        .collect::<Vec<_>>();

    store.mset(key_value_pairs.clone());
    assert_eq!(store.mget(&keys), expected_values);
    assert_yielded_keys(store, &expected_keys, prefix);

    store.amset(key_value_pairs).await;
    assert_eq!(store.amget(keys.clone()).await, expected_values);
    assert_async_yielded_keys(store, &expected_keys, prefix).await;

    store.amdelete(keys.clone()).await;
    assert!(store.amget(keys).await.iter().all(Option::is_none));
}
