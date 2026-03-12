use std::fs;
use std::path::{Path, PathBuf};

use langchain_core::LangChainError;

use super::types::AgentMiddleware;

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

impl AgentMiddleware for FilesystemFileSearchMiddleware {}

fn should_skip_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| matches!(name, ".git" | "target" | "node_modules"))
}

fn io_error(error: std::io::Error) -> LangChainError {
    LangChainError::request(error.to_string())
}
