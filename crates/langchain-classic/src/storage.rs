use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use langchain_core::LangChainError;
use langchain_core::documents::Document;
pub use langchain_core::stores::*;
use serde::Serialize;
use serde::de::DeserializeOwned;

pub type InvalidKeyException = LangChainError;

type Encoder<V> = Arc<dyn Fn(&V) -> Result<Vec<u8>, LangChainError> + Send + Sync>;
type Decoder<V> = Arc<dyn Fn(&[u8]) -> Result<V, LangChainError> + Send + Sync>;

pub struct EncoderBackedStore<S, V> {
    store: S,
    encoder: Encoder<V>,
    decoder: Decoder<V>,
}

impl<S, V> EncoderBackedStore<S, V> {
    pub fn new(store: S, encoder: Encoder<V>, decoder: Decoder<V>) -> Self {
        Self {
            store,
            encoder,
            decoder,
        }
    }

    pub fn inner(&self) -> &S {
        &self.store
    }
}

impl<S, V> fmt::Debug for EncoderBackedStore<S, V>
where
    S: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EncoderBackedStore")
            .field("store", &self.store)
            .finish_non_exhaustive()
    }
}

impl<S, V> BaseStore<V> for EncoderBackedStore<S, V>
where
    S: BaseStore<Vec<u8>>,
    V: Clone + Send + Sync + 'static,
{
    fn mget(&self, keys: &[String]) -> Vec<Option<V>> {
        self.store
            .mget(keys)
            .into_iter()
            .map(|value| value.and_then(|bytes| (self.decoder)(&bytes).ok()))
            .collect()
    }

    fn mset(&self, key_value_pairs: Vec<(String, V)>) {
        self.store.mset(
            key_value_pairs
                .into_iter()
                .map(|(key, value)| {
                    let bytes = (self.encoder)(&value)
                        .expect("encoder-backed store failed to serialize value");
                    (key, bytes)
                })
                .collect(),
        );
    }

    fn mdelete(&self, keys: &[String]) {
        self.store.mdelete(keys);
    }

    fn yield_keys<'a>(&'a self, prefix: Option<&'a str>) -> Box<dyn Iterator<Item = String> + 'a> {
        self.store.yield_keys(prefix)
    }
}

#[derive(Debug, Clone)]
pub struct LocalFileStore {
    root: Arc<PathBuf>,
}

impl LocalFileStore {
    pub fn new(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).expect("local file store root should be creatable");
        Self {
            root: Arc::new(root),
        }
    }

    fn path_for_key(&self, key: &str) -> PathBuf {
        let mut path = (*self.root).clone();
        for component in Path::new(key).components() {
            match component {
                Component::Normal(segment) => path.push(segment),
                _ => panic!("invalid storage key `{key}`"),
            }
        }
        path
    }

    fn walk_keys(root: &Path, current: &Path, keys: &mut Vec<String>) {
        let Ok(entries) = fs::read_dir(current) else {
            return;
        };

        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                Self::walk_keys(root, &path, keys);
            } else if let Ok(relative) = path.strip_prefix(root) {
                keys.push(relative.to_string_lossy().replace('\\', "/"));
            }
        }
    }
}

impl BaseStore<Vec<u8>> for LocalFileStore {
    fn mget(&self, keys: &[String]) -> Vec<Option<Vec<u8>>> {
        keys.iter()
            .map(|key| fs::read(self.path_for_key(key)).ok())
            .collect()
    }

    fn mset(&self, key_value_pairs: Vec<(String, Vec<u8>)>) {
        for (key, value) in key_value_pairs {
            let path = self.path_for_key(&key);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("local file store parent should exist");
            }
            fs::write(path, value).expect("local file store should write bytes");
        }
    }

    fn mdelete(&self, keys: &[String]) {
        for key in keys {
            let path = self.path_for_key(key);
            if path.exists() {
                let _ = fs::remove_file(path);
            }
        }
    }

    fn yield_keys<'a>(&'a self, prefix: Option<&'a str>) -> Box<dyn Iterator<Item = String> + 'a> {
        let mut keys = Vec::new();
        Self::walk_keys(&self.root, &self.root, &mut keys);
        keys.sort();
        Box::new(
            keys.into_iter()
                .filter(move |key| prefix.is_none_or(|prefix| key.starts_with(prefix))),
        )
    }
}

pub fn create_lc_store<V, S>(store: S) -> EncoderBackedStore<S, V>
where
    V: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
    S: BaseStore<Vec<u8>>,
{
    EncoderBackedStore::new(
        store,
        Arc::new(|value| serde_json::to_vec(value).map_err(Into::into)),
        Arc::new(|value| serde_json::from_slice(value).map_err(Into::into)),
    )
}

pub fn create_kv_docstore<S>(store: S) -> EncoderBackedStore<S, Document>
where
    S: BaseStore<Vec<u8>>,
{
    create_lc_store(store)
}
