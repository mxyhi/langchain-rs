use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::RwLock;

use futures_util::future::BoxFuture;

use crate::LangChainError;
use crate::outputs::GenerationCandidate;

pub type CacheValue = Vec<GenerationCandidate>;

pub trait BaseCache: Send + Sync {
    fn lookup(&self, prompt: &str, llm_string: &str) -> Option<CacheValue>;

    fn update(&self, prompt: &str, llm_string: &str, return_val: CacheValue);

    fn clear(&self);

    fn alookup<'a>(
        &'a self,
        prompt: &'a str,
        llm_string: &'a str,
    ) -> BoxFuture<'a, Option<CacheValue>> {
        Box::pin(async move { self.lookup(prompt, llm_string) })
    }

    fn aupdate<'a>(
        &'a self,
        prompt: &'a str,
        llm_string: &'a str,
        return_val: CacheValue,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move { self.update(prompt, llm_string, return_val) })
    }

    fn aclear<'a>(&'a self) -> BoxFuture<'a, ()> {
        Box::pin(async move { self.clear() })
    }
}

#[derive(Default)]
struct CacheState {
    entries: HashMap<(String, String), CacheValue>,
    order: VecDeque<(String, String)>,
}

pub struct InMemoryCache {
    state: RwLock<CacheState>,
    maxsize: Option<usize>,
}

impl InMemoryCache {
    pub fn new() -> Self {
        Self {
            state: RwLock::new(CacheState::default()),
            maxsize: None,
        }
    }

    pub fn with_maxsize(maxsize: usize) -> Result<Self, LangChainError> {
        if maxsize == 0 {
            return Err(LangChainError::request(
                "cache maxsize must be greater than 0",
            ));
        }

        Ok(Self {
            state: RwLock::new(CacheState::default()),
            maxsize: Some(maxsize),
        })
    }
}

impl Default for InMemoryCache {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for InMemoryCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InMemoryCache")
            .field("maxsize", &self.maxsize)
            .finish_non_exhaustive()
    }
}

impl BaseCache for InMemoryCache {
    fn lookup(&self, prompt: &str, llm_string: &str) -> Option<CacheValue> {
        self.state
            .read()
            .expect("cache read lock poisoned")
            .entries
            .get(&(prompt.to_owned(), llm_string.to_owned()))
            .cloned()
    }

    fn update(&self, prompt: &str, llm_string: &str, return_val: CacheValue) {
        let key = (prompt.to_owned(), llm_string.to_owned());
        let mut state = self.state.write().expect("cache write lock poisoned");

        if !state.entries.contains_key(&key) {
            state.order.push_back(key.clone());
        }

        state.entries.insert(key.clone(), return_val);

        if let Some(maxsize) = self.maxsize {
            while state.entries.len() > maxsize {
                if let Some(oldest) = state.order.pop_front() {
                    state.entries.remove(&oldest);
                }
            }
        }
    }

    fn clear(&self) {
        let mut state = self.state.write().expect("cache write lock poisoned");
        state.entries.clear();
        state.order.clear();
    }
}
