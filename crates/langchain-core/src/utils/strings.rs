use serde_json::{Map, Value};

pub fn stringify_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Object(map) => format!("\n{}", stringify_dict(map)),
        Value::Array(items) => items
            .iter()
            .map(stringify_value)
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => "null".to_owned(),
        _ => value.to_string(),
    }
}

pub fn stringify_dict(data: &Map<String, Value>) -> String {
    data.iter()
        .map(|(key, value)| format!("{key}: {}\n", stringify_value(value)))
        .collect()
}

pub fn comma_list<T>(items: impl IntoIterator<Item = T>) -> String
where
    T: std::fmt::Display,
{
    items
        .into_iter()
        .map(|item| item.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn sanitize_for_postgres(text: &str, replacement: &str) -> String {
    text.replace('\0', replacement)
}
