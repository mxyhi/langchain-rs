use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use crate::caches::BaseCache;

static DEBUG: AtomicBool = AtomicBool::new(false);
static VERBOSE: AtomicBool = AtomicBool::new(false);
static LLM_CACHE: OnceLock<RwLock<Option<Arc<dyn BaseCache>>>> = OnceLock::new();

fn llm_cache_cell() -> &'static RwLock<Option<Arc<dyn BaseCache>>> {
    LLM_CACHE.get_or_init(|| RwLock::new(None))
}

pub fn set_debug(value: bool) {
    DEBUG.store(value, Ordering::SeqCst);
}

pub fn get_debug() -> bool {
    DEBUG.load(Ordering::SeqCst)
}

pub fn set_verbose(value: bool) {
    VERBOSE.store(value, Ordering::SeqCst);
}

pub fn get_verbose() -> bool {
    VERBOSE.load(Ordering::SeqCst)
}

pub fn set_llm_cache(value: Option<Arc<dyn BaseCache>>) {
    *llm_cache_cell()
        .write()
        .expect("llm cache write lock poisoned") = value;
}

pub fn get_llm_cache() -> Option<Arc<dyn BaseCache>> {
    llm_cache_cell()
        .read()
        .expect("llm cache read lock poisoned")
        .clone()
}
