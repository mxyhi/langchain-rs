use serde_json::{Map, Number, Value};

use crate::LangChainError;

pub fn add_usage(
    left: &Map<String, Value>,
    right: &Map<String, Value>,
) -> Result<Map<String, Value>, LangChainError> {
    dict_int_op(left, right, |left, right| left + right)
}

pub fn subtract_usage(
    left: &Map<String, Value>,
    right: &Map<String, Value>,
) -> Result<Map<String, Value>, LangChainError> {
    dict_int_op(left, right, |left, right| left - right)
}

pub fn dict_int_op(
    left: &Map<String, Value>,
    right: &Map<String, Value>,
    operation: impl Fn(i64, i64) -> i64 + Copy,
) -> Result<Map<String, Value>, LangChainError> {
    let mut combined = Map::new();

    for key in left.keys().chain(right.keys()) {
        if combined.contains_key(key) {
            continue;
        }

        let left_value = left.get(key);
        let right_value = right.get(key);

        let combined_value = match (left_value, right_value) {
            (Some(Value::Number(left)), Some(Value::Number(right))) => {
                Value::Number(Number::from(operation(
                    left.as_i64().ok_or_else(|| {
                        LangChainError::unsupported(format!(
                            "usage value for `{key}` must be a signed integer"
                        ))
                    })?,
                    right.as_i64().ok_or_else(|| {
                        LangChainError::unsupported(format!(
                            "usage value for `{key}` must be a signed integer"
                        ))
                    })?,
                )))
            }
            (Some(Value::Number(left)), None) => Value::Number(Number::from(operation(
                left.as_i64().ok_or_else(|| {
                    LangChainError::unsupported(format!(
                        "usage value for `{key}` must be a signed integer"
                    ))
                })?,
                0,
            ))),
            (None, Some(Value::Number(right))) => Value::Number(Number::from(operation(
                0,
                right.as_i64().ok_or_else(|| {
                    LangChainError::unsupported(format!(
                        "usage value for `{key}` must be a signed integer"
                    ))
                })?,
            ))),
            (Some(Value::Object(left)), Some(Value::Object(right))) => {
                Value::Object(dict_int_op(left, right, operation)?)
            }
            (Some(Value::Object(left)), None) => {
                Value::Object(dict_int_op(left, &Map::new(), operation)?)
            }
            (None, Some(Value::Object(right))) => {
                Value::Object(dict_int_op(&Map::new(), right, operation)?)
            }
            _ => {
                return Err(LangChainError::unsupported(format!(
                    "Unknown usage value types for `{key}`. Only nested objects and signed integers are supported."
                )));
            }
        };

        combined.insert(key.clone(), combined_value);
    }

    Ok(combined)
}

pub fn _dict_int_op(
    left: &Value,
    right: &Value,
    operation: impl Fn(i64, i64) -> i64 + Copy,
    _default: i64,
    depth: usize,
    max_depth: usize,
) -> Result<Value, LangChainError> {
    if depth >= max_depth {
        return Err(LangChainError::unsupported(format!(
            "{max_depth} exceeded, unable to combine usage payloads."
        )));
    }

    match (left, right) {
        (Value::Object(left), Value::Object(right)) => {
            Ok(Value::Object(dict_int_op(left, right, operation)?))
        }
        _ => Err(LangChainError::unsupported(
            "_dict_int_op only supports object roots".to_owned(),
        )),
    }
}
