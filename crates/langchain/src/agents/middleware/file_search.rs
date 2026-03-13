use std::fs;
use std::path::{Path, PathBuf};

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::messages::{ToolMessage, ToolMessageStatus};
use serde_json::{Value, json};

use super::types::{AgentMiddleware, ToolCallHandler, ToolCallRequest};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSearchHit {
    path: PathBuf,
    line_number: usize,
    line: String,
}

impl FileSearchHit {
    pub fn new(path: PathBuf, line_number: usize, line: impl Into<String>) -> Self {
        Self {
            path,
            line_number,
            line: line.into(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn line_number(&self) -> usize {
        self.line_number
    }

    pub fn line(&self) -> &str {
        &self.line
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilesystemFileSearchMiddleware {
    root: PathBuf,
    max_results: usize,
    max_file_bytes: u64,
}

impl FilesystemFileSearchMiddleware {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            max_results: 20,
            max_file_bytes: 256 * 1024,
        }
    }

    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results.max(1);
        self
    }

    pub fn with_max_file_bytes(mut self, max_file_bytes: u64) -> Self {
        self.max_file_bytes = max_file_bytes.max(1);
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn search(&self, query: &str) -> Result<Vec<FileSearchHit>, LangChainError> {
        self.search_with_limit(query, self.max_results)
    }

    pub fn search_with_limit(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<FileSearchHit>, LangChainError> {
        let normalized_query = query.trim().to_ascii_lowercase();
        if normalized_query.is_empty() {
            return Ok(Vec::new());
        }

        let mut hits = Vec::new();
        self.walk(&self.root, &normalized_query, limit.max(1), &mut hits)?;
        Ok(hits)
    }

    fn walk(
        &self,
        current: &Path,
        query: &str,
        limit: usize,
        hits: &mut Vec<FileSearchHit>,
    ) -> Result<(), LangChainError> {
        if hits.len() >= limit {
            return Ok(());
        }

        let metadata = fs::metadata(current).map_err(io_error)?;
        if metadata.is_file() {
            self.search_file(current, query, limit, hits)?;
            return Ok(());
        }

        let entries = fs::read_dir(current).map_err(io_error)?;
        for entry in entries {
            let entry = entry.map_err(io_error)?;
            let path = entry.path();
            if should_skip_path(&path) {
                continue;
            }
            self.walk(&path, query, limit, hits)?;
            if hits.len() >= limit {
                break;
            }
        }
        Ok(())
    }

    fn search_file(
        &self,
        path: &Path,
        query: &str,
        limit: usize,
        hits: &mut Vec<FileSearchHit>,
    ) -> Result<(), LangChainError> {
        let metadata = fs::metadata(path).map_err(io_error)?;
        if metadata.len() > self.max_file_bytes {
            return Ok(());
        }

        let Ok(contents) = fs::read_to_string(path) else {
            return Ok(());
        };

        for (index, line) in contents.lines().enumerate() {
            if line.to_ascii_lowercase().contains(query) {
                hits.push(FileSearchHit::new(path.to_path_buf(), index + 1, line));
                if hits.len() >= limit {
                    break;
                }
            }
        }
        Ok(())
    }
}

impl AgentMiddleware for FilesystemFileSearchMiddleware {
    fn wrap_tool_call(
        &self,
        request: ToolCallRequest,
        handler: ToolCallHandler,
    ) -> BoxFuture<'static, Result<ToolMessage, LangChainError>> {
        if !matches!(request.tool_call().name(), "file_search" | "grep_search") {
            return handler(request);
        }

        let middleware = self.clone();
        Box::pin(async move {
            let query = extract_query(request.tool_call().args())?;
            let limit = extract_limit(request.tool_call().args());
            let hits = middleware.search_with_limit(query, limit)?;
            let rendered_hits = hits
                .iter()
                .map(|hit| {
                    format!(
                        "{}:{}:{}",
                        display_path(&middleware.root, hit.path()),
                        hit.line_number(),
                        hit.line()
                    )
                })
                .collect::<Vec<_>>();
            let content = if rendered_hits.is_empty() {
                "No matches found".to_owned()
            } else {
                rendered_hits.join("\n")
            };
            let artifact = hits
                .iter()
                .map(|hit| {
                    json!({
                        "path": display_path(&middleware.root, hit.path()),
                        "line_number": hit.line_number(),
                        "line": hit.line(),
                    })
                })
                .collect::<Vec<_>>();
            Ok(ToolMessage::with_parts(
                content,
                request.tool_call().id().unwrap_or_default(),
                Some(request.tool_call().name()),
                Some(Value::Array(artifact)),
                ToolMessageStatus::Success,
            ))
        })
    }
}

fn extract_query(args: &Value) -> Result<&str, LangChainError> {
    match args {
        Value::String(query) if !query.trim().is_empty() => Ok(query),
        Value::Object(map) => ["query", "input", "pattern"]
            .iter()
            .find_map(|key| map.get(*key).and_then(Value::as_str))
            .filter(|query| !query.trim().is_empty())
            .ok_or_else(|| {
                LangChainError::request(
                    "file_search tool call requires a non-empty `query`, `input`, or `pattern`",
                )
            }),
        _ => Err(LangChainError::request(
            "file_search tool call requires string arguments",
        )),
    }
}

fn extract_limit(args: &Value) -> usize {
    match args {
        Value::Object(map) => map
            .get("limit")
            .and_then(Value::as_u64)
            .and_then(|limit| usize::try_from(limit).ok())
            .filter(|limit| *limit > 0)
            .unwrap_or(20),
        _ => 20,
    }
}

fn display_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .map(|relative| format!("/{}", relative.display()))
        .unwrap_or_else(|_| path.display().to_string())
}

fn should_skip_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| matches!(name, ".git" | "target" | "node_modules"))
}

fn io_error(error: std::io::Error) -> LangChainError {
    LangChainError::request(error.to_string())
}
