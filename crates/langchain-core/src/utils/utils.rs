use std::collections::{BTreeMap, BTreeSet};

use secrecy::SecretString;
use serde_json::Value;

use crate::LangChainError;

use super::uuid::uuid7;

pub fn build_extra_kwargs(
    explicit: BTreeMap<String, Value>,
    values: BTreeMap<String, Value>,
    reserved: BTreeSet<String>,
) -> Result<BTreeMap<String, Value>, LangChainError> {
    let mut merged = explicit.clone();

    for (key, value) in values {
        if reserved.contains(&key) {
            continue;
        }

        match merged.get(&key) {
            Some(existing) if existing != &value => {
                return Err(LangChainError::unsupported(format!(
                    "duplicate extra kwarg `{key}` with conflicting values"
                )));
            }
            Some(_) => {}
            None => {
                merged.insert(key, value);
            }
        }
    }

    Ok(merged)
}

pub fn ensure_id(id: Option<String>) -> String {
    id.unwrap_or_else(|| format!("lc_{}", uuid7().simple()))
}

pub fn from_env(
    env_key: impl Into<String>,
    default: Option<String>,
    _description: Option<&str>,
) -> impl Fn() -> Result<String, LangChainError> {
    let env_key = env_key.into();
    move || match std::env::var(&env_key) {
        Ok(value) if !value.is_empty() => Ok(value),
        _ => default.clone().ok_or_else(|| {
            LangChainError::unsupported(format!(
                "environment variable `{env_key}` is required but was not set"
            ))
        }),
    }
}

pub fn secret_from_env(
    env_key: impl Into<String>,
    default: Option<String>,
    description: Option<&str>,
) -> impl Fn() -> Result<SecretString, LangChainError> {
    let loader = from_env(env_key, default, description);
    move || loader().map(|value| SecretString::new(value.into()))
}
