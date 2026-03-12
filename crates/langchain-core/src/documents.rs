use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub page_content: String,
    pub metadata: BTreeMap<String, Value>,
    pub id: Option<String>,
}

impl Document {
    pub fn new(page_content: impl Into<String>) -> Self {
        Self {
            page_content: page_content.into(),
            metadata: BTreeMap::new(),
            id: None,
        }
    }
}
