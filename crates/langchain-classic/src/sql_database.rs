use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use langchain_core::LangChainError;
use rusqlite::Connection;
use rusqlite::types::ValueRef;
use serde_json::Value;

fn map_sql_error(error: rusqlite::Error) -> LangChainError {
    LangChainError::request(error.to_string())
}

fn sqlite_value_to_json(value: ValueRef<'_>) -> Value {
    match value {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(value) => Value::from(value),
        ValueRef::Real(value) => Value::from(value),
        ValueRef::Text(value) => Value::from(String::from_utf8_lossy(value).to_string()),
        ValueRef::Blob(value) => Value::Array(value.iter().copied().map(Value::from).collect()),
    }
}

#[derive(Debug, Clone)]
pub struct SQLDatabase {
    path: PathBuf,
}

impl SQLDatabase {
    pub fn from_sqlite_path(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn connect(&self) -> Result<Connection, LangChainError> {
        Connection::open(&self.path).map_err(map_sql_error)
    }

    pub fn execute_batch(&self, sql: &str) -> Result<(), LangChainError> {
        self.connect()?.execute_batch(sql).map_err(map_sql_error)
    }

    pub fn query(&self, sql: &str) -> Result<Vec<BTreeMap<String, Value>>, LangChainError> {
        let connection = self.connect()?;
        let mut statement = connection.prepare(sql).map_err(map_sql_error)?;
        let column_names = statement
            .column_names()
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        let mut rows = statement.query([]).map_err(map_sql_error)?;
        let mut collected = Vec::new();

        while let Some(row) = rows.next().map_err(map_sql_error)? {
            let mut object = BTreeMap::new();
            for (index, column_name) in column_names.iter().enumerate() {
                let value = row.get_ref(index).map_err(map_sql_error)?;
                object.insert(column_name.clone(), sqlite_value_to_json(value));
            }
            collected.push(object);
        }

        Ok(collected)
    }

    pub fn get_usable_table_names(&self) -> Result<Vec<String>, LangChainError> {
        Ok(self
            .query(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
            )?
            .into_iter()
            .filter_map(|row| row.get("name").and_then(Value::as_str).map(str::to_owned))
            .collect())
    }
}
