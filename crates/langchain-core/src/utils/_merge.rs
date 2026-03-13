use serde_json::{Map, Number, Value};

use crate::LangChainError;

pub fn merge_dicts<'a, I>(
    left: &Map<String, Value>,
    others: I,
) -> Result<Map<String, Value>, LangChainError>
where
    I: IntoIterator<Item = &'a Map<String, Value>>,
{
    let mut merged = left.clone();

    for right in others {
        for (key, right_value) in right {
            match merged.get(key) {
                None => {
                    merged.insert(key.clone(), right_value.clone());
                }
                Some(left_value) if left_value.is_null() && !right_value.is_null() => {
                    merged.insert(key.clone(), right_value.clone());
                }
                Some(_) if right_value.is_null() => {}
                Some(left_value) => {
                    let merged_value = merge_field(key, left_value, right_value)?;
                    merged.insert(key.clone(), merged_value);
                }
            }
        }
    }

    Ok(merged)
}

pub fn merge_lists<'a, I>(left: &[Value], others: I) -> Result<Vec<Value>, LangChainError>
where
    I: IntoIterator<Item = &'a Vec<Value>>,
{
    let mut merged = left.to_vec();

    for other in others {
        for element in other {
            if let Some(position) = find_merge_target(&merged, element) {
                merged[position] = merge_obj(&merged[position], element)?;
            } else {
                merged.push(element.clone());
            }
        }
    }

    Ok(merged)
}

pub fn merge_obj(left: &Value, right: &Value) -> Result<Value, LangChainError> {
    if left.is_null() || right.is_null() {
        return Ok(if left.is_null() {
            right.clone()
        } else {
            left.clone()
        });
    }

    match (left, right) {
        (Value::String(left), Value::String(right)) => Ok(Value::String(format!("{left}{right}"))),
        (Value::Object(left), Value::Object(right)) => {
            Ok(Value::Object(merge_dicts(left, [right])?))
        }
        (Value::Array(left), Value::Array(right)) => Ok(Value::Array(merge_lists(left, [right])?)),
        (Value::Bool(left), Value::Bool(right)) if left == right => Ok(Value::Bool(*left)),
        (Value::Number(left), Value::Number(right)) => Ok(Value::Number(add_numbers(left, right)?)),
        _ if left == right => Ok(left.clone()),
        _ => Err(LangChainError::unsupported(format!(
            "Unable to merge values `{left}` and `{right}`."
        ))),
    }
}

fn merge_field(key: &str, left: &Value, right: &Value) -> Result<Value, LangChainError> {
    if matches!(key, "id" | "output_version" | "model_provider" | "index") && left == right {
        return Ok(left.clone());
    }

    match (left, right) {
        (Value::String(left), Value::String(right)) => Ok(Value::String(format!("{left}{right}"))),
        (Value::Object(left), Value::Object(right)) => {
            Ok(Value::Object(merge_dicts(left, [right])?))
        }
        (Value::Array(left), Value::Array(right)) => Ok(Value::Array(merge_lists(left, [right])?)),
        (Value::Number(left), Value::Number(right)) => Ok(Value::Number(add_numbers(left, right)?)),
        _ if left == right => Ok(left.clone()),
        _ => Err(LangChainError::unsupported(format!(
            "additional_kwargs[\"{key}\"] already exists with incompatible values `{left}` and `{right}`"
        ))),
    }
}

fn add_numbers(left: &Number, right: &Number) -> Result<Number, LangChainError> {
    match (left.as_i64(), right.as_i64()) {
        (Some(left), Some(right)) => Ok(Number::from(left + right)),
        _ => match (left.as_u64(), right.as_u64()) {
            (Some(left), Some(right)) => Ok(Number::from(left + right)),
            _ => Err(LangChainError::unsupported(format!(
                "cannot merge non-integer numbers `{left}` and `{right}`"
            ))),
        },
    }
}

fn find_merge_target(existing: &[Value], candidate: &Value) -> Option<usize> {
    let candidate = candidate.as_object()?;
    let candidate_index = candidate.get("index")?;
    let candidate_id = candidate.get("id");

    existing.iter().position(|current| {
        let Some(current) = current.as_object() else {
            return false;
        };

        if current.get("index") != Some(candidate_index) {
            return false;
        }

        let current_id = current.get("id");
        match (current_id, candidate_id) {
            (None, _) | (_, None) => true,
            (Some(Value::String(current)), Some(Value::String(candidate))) => {
                current.is_empty() || candidate.is_empty() || current == candidate
            }
            (Some(current), Some(candidate)) => current == candidate,
        }
    })
}
