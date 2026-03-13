use std::collections::BTreeMap;

use crate::LangChainError;

pub fn env_var_is_set(env_var: &str) -> bool {
    std::env::var(env_var)
        .map(|value| !matches!(value.as_str(), "" | "0" | "false" | "False"))
        .unwrap_or(false)
}

pub fn get_from_dict_or_env<K, I>(
    data: &BTreeMap<String, String>,
    keys: I,
    env_key: &str,
    default: Option<&str>,
) -> Result<String, LangChainError>
where
    I: IntoIterator<Item = K>,
    K: AsRef<str>,
{
    let keys = keys
        .into_iter()
        .map(|key| key.as_ref().to_owned())
        .collect::<Vec<_>>();

    for key in &keys {
        if let Some(value) = data.get(key).filter(|value| !value.is_empty()) {
            return Ok(value.clone());
        }
    }

    let key_for_error = keys.first().cloned().unwrap_or_else(|| "value".to_owned());
    get_from_env(&key_for_error, env_key, default)
}

pub fn get_from_env(
    key: &str,
    env_key: &str,
    default: Option<&str>,
) -> Result<String, LangChainError> {
    if let Ok(value) = std::env::var(env_key) {
        if !value.is_empty() {
            return Ok(value);
        }
    }

    if let Some(default) = default {
        return Ok(default.to_owned());
    }

    Err(LangChainError::unsupported(format!(
        "Did not find {key}, please add environment variable `{env_key}` or pass `{key}` directly."
    )))
}
