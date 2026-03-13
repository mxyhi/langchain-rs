use std::collections::BTreeMap;

use langchain_core::LangChainError;
use langchain_core::tools::{ToolDefinition, tool};

use super::anthropic_tools::{FileData, normalize_virtual_path};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrepOutputMode {
    FilesWithMatches,
    Content,
    Count,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrepHit {
    pub path: String,
    pub line_number: Option<usize>,
    pub line: Option<String>,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateFileSearchMiddleware {
    max_results: usize,
}

impl Default for StateFileSearchMiddleware {
    fn default() -> Self {
        Self { max_results: 20 }
    }
}

impl StateFileSearchMiddleware {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results.max(1);
        self
    }

    pub fn tools(&self) -> Vec<ToolDefinition> {
        vec![
            tool("glob_search", "Find virtual files by glob pattern").with_parameters(
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string" },
                        "path": { "type": "string" }
                    },
                    "required": ["pattern"]
                }),
            ),
            tool("grep_search", "Search file contents by literal pattern").with_parameters(
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string" },
                        "path": { "type": "string" },
                        "include": { "type": "string" },
                        "output_mode": { "type": "string" }
                    },
                    "required": ["pattern"]
                }),
            ),
        ]
    }

    pub fn glob_search(
        &self,
        pattern: &str,
        path: &str,
        files: &BTreeMap<String, FileData>,
    ) -> Result<Vec<String>, LangChainError> {
        let base_path = normalize_virtual_path(path, &[])?;
        let mut matches = files
            .keys()
            .filter_map(|file_path| relative_path(file_path, &base_path))
            .filter(|relative| glob_matches(pattern, relative))
            .map(|relative| {
                format!("{}/{}", base_path.trim_end_matches('/'), relative).replace("//", "/")
            })
            .collect::<Vec<_>>();

        if base_path == "/" {
            matches = files
                .keys()
                .filter(|file_path| glob_matches(pattern, file_path.trim_start_matches('/')))
                .cloned()
                .collect();
        }

        matches.sort();
        matches.truncate(self.max_results);
        Ok(matches)
    }

    pub fn grep_search(
        &self,
        pattern: &str,
        path: &str,
        include: Option<&str>,
        output_mode: GrepOutputMode,
        files: &BTreeMap<String, FileData>,
    ) -> Result<Vec<GrepHit>, LangChainError> {
        let base_path = normalize_virtual_path(path, &[])?;
        let mut hits = Vec::new();

        for (file_path, file_data) in files {
            let Some(relative) = relative_path(file_path, &base_path) else {
                continue;
            };
            if include.is_some_and(|pattern| !include_matches(pattern, relative)) {
                continue;
            }

            let matching_lines = file_data
                .lines()
                .iter()
                .enumerate()
                .filter(|(_, line)| line.contains(pattern))
                .collect::<Vec<_>>();
            if matching_lines.is_empty() {
                continue;
            }

            match output_mode {
                GrepOutputMode::FilesWithMatches => hits.push(GrepHit {
                    path: file_path.clone(),
                    line_number: None,
                    line: None,
                    count: matching_lines.len(),
                }),
                GrepOutputMode::Count => hits.push(GrepHit {
                    path: file_path.clone(),
                    line_number: None,
                    line: None,
                    count: matching_lines.len(),
                }),
                GrepOutputMode::Content => {
                    hits.extend(matching_lines.into_iter().map(|(index, line)| GrepHit {
                        path: file_path.clone(),
                        line_number: Some(index + 1),
                        line: Some(line.clone()),
                        count: 1,
                    }));
                }
            }

            if hits.len() >= self.max_results {
                break;
            }
        }

        hits.truncate(self.max_results);
        Ok(hits)
    }
}

fn relative_path<'a>(file_path: &'a str, base_path: &str) -> Option<&'a str> {
    if base_path == "/" {
        return Some(file_path.trim_start_matches('/'));
    }
    if file_path == base_path {
        return Some("");
    }
    file_path
        .strip_prefix(&format!("{base_path}/"))
        .or_else(|| file_path.strip_prefix(base_path))
}

fn glob_matches(pattern: &str, candidate: &str) -> bool {
    glob_matches_bytes(pattern.as_bytes(), candidate.as_bytes())
}

fn include_matches(pattern: &str, relative: &str) -> bool {
    glob_matches(pattern, relative)
        || relative
            .rsplit('/')
            .next()
            .is_some_and(|file_name| glob_matches(pattern, file_name))
}

fn glob_matches_bytes(pattern: &[u8], candidate: &[u8]) -> bool {
    if pattern.is_empty() {
        return candidate.is_empty();
    }

    if pattern.starts_with(b"**") {
        let rest = &pattern[2..];
        return glob_matches_bytes(rest, candidate)
            || (!candidate.is_empty() && glob_matches_bytes(pattern, &candidate[1..]));
    }

    match pattern[0] {
        b'*' => {
            glob_matches_bytes(&pattern[1..], candidate)
                || (!candidate.is_empty()
                    && candidate[0] != b'/'
                    && glob_matches_bytes(pattern, &candidate[1..]))
        }
        b'?' => {
            !candidate.is_empty()
                && candidate[0] != b'/'
                && glob_matches_bytes(&pattern[1..], &candidate[1..])
        }
        character => {
            !candidate.is_empty()
                && character == candidate[0]
                && glob_matches_bytes(&pattern[1..], &candidate[1..])
        }
    }
}
