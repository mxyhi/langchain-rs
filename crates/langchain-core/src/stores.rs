use std::collections::HashMap;
use std::fmt;
use std::sync::RwLock;

use futures_util::future::BoxFuture;
use futures_util::stream::{self, BoxStream, StreamExt};

pub trait BaseStore<V>: Send + Sync
where
    V: Clone + Send + Sync + 'static,
{
    fn mget(&self, keys: &[String]) -> Vec<Option<V>>;

    fn mset(&self, key_value_pairs: Vec<(String, V)>);

    fn mdelete(&self, keys: &[String]);

    fn yield_keys<'a>(&'a self, prefix: Option<&'a str>) -> Box<dyn Iterator<Item = String> + 'a>;

    fn amget<'a>(&'a self, keys: Vec<String>) -> BoxFuture<'a, Vec<Option<V>>> {
        Box::pin(async move { self.mget(&keys) })
    }

    fn amset<'a>(&'a self, key_value_pairs: Vec<(String, V)>) -> BoxFuture<'a, ()> {
        Box::pin(async move { self.mset(key_value_pairs) })
    }

    fn amdelete<'a>(&'a self, keys: Vec<String>) -> BoxFuture<'a, ()> {
        Box::pin(async move { self.mdelete(&keys) })
    }

    fn ayield_keys<'a>(&'a self, prefix: Option<&'a str>) -> BoxStream<'a, String> {
        stream::iter(self.yield_keys(prefix).collect::<Vec<_>>()).boxed()
    }
}

pub struct InMemoryBaseStore<V> {
    store: RwLock<HashMap<String, V>>,
}

impl<V> InMemoryBaseStore<V> {
    pub fn new() -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
        }
    }
}

impl<V> Default for InMemoryBaseStore<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> fmt::Debug for InMemoryBaseStore<V> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InMemoryBaseStore")
            .finish_non_exhaustive()
    }
}

impl<V> BaseStore<V> for InMemoryBaseStore<V>
where
    V: Clone + Send + Sync + 'static,
{
    fn mget(&self, keys: &[String]) -> Vec<Option<V>> {
        let store = self.store.read().expect("store read lock poisoned");
        keys.iter().map(|key| store.get(key).cloned()).collect()
    }

    fn mset(&self, key_value_pairs: Vec<(String, V)>) {
        let mut store = self.store.write().expect("store write lock poisoned");
        for (key, value) in key_value_pairs {
            store.insert(key, value);
        }
    }

    fn mdelete(&self, keys: &[String]) {
        let mut store = self.store.write().expect("store write lock poisoned");
        for key in keys {
            store.remove(key);
        }
    }

    fn yield_keys<'a>(&'a self, prefix: Option<&'a str>) -> Box<dyn Iterator<Item = String> + 'a> {
        let keys = self
            .store
            .read()
            .expect("store read lock poisoned")
            .keys()
            .filter(|key| prefix.is_none_or(|prefix| key.starts_with(prefix)))
            .cloned()
            .collect::<Vec<_>>();
        Box::new(keys.into_iter())
    }
}

pub type InMemoryStore<V> = InMemoryBaseStore<V>;
pub type InMemoryByteStore = InMemoryBaseStore<Vec<u8>>;
