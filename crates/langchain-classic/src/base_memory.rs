use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use serde_json::Value;

pub trait BaseMemory: Send + Sync {
    fn memory_variables(&self) -> Vec<String>;

    fn load_memory_variables(&self, inputs: BTreeMap<String, Value>) -> BTreeMap<String, Value>;

    fn save_context(&self, inputs: BTreeMap<String, Value>, outputs: BTreeMap<String, Value>);

    fn clear(&self);

    fn aload_memory_variables<'a>(
        &'a self,
        inputs: BTreeMap<String, Value>,
    ) -> BoxFuture<'a, BTreeMap<String, Value>> {
        Box::pin(async move { self.load_memory_variables(inputs) })
    }

    fn asave_context<'a>(
        &'a self,
        inputs: BTreeMap<String, Value>,
        outputs: BTreeMap<String, Value>,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move { self.save_context(inputs, outputs) })
    }

    fn aclear<'a>(&'a self) -> BoxFuture<'a, ()> {
        Box::pin(async move { self.clear() })
    }
}
