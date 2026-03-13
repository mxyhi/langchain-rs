use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use langchain_core::LangChainError;
use serde::{Deserialize, Serialize};

pub const TEXT_EDITOR_TOOL_TYPE: &str = "text_editor_20250728";
pub const TEXT_EDITOR_TOOL_NAME: &str = "str_replace_based_edit_tool";
pub const MEMORY_TOOL_TYPE: &str = "memory_20250818";
pub const MEMORY_TOOL_NAME: &str = "memory";
pub const MEMORY_SYSTEM_PROMPT: &str = "IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE.\nMEMORY PROTOCOL:\n1. Use the `view` command of your `memory` tool to check for earlier progress.\n2. Continue the task and record status and conclusions in memory files.\n3. Assume interruption at any moment and persist anything important before moving on.";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicBuiltInTool {
    #[serde(rename = "type")]
    kind: String,
    name: String,
}

impl AnthropicBuiltInTool {
    pub fn new(kind: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            name: name.into(),
        }
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileData {
    content: Vec<String>,
    created_at: String,
    modified_at: String,
}

impl FileData {
    pub fn new(text: &str) -> Self {
        let timestamp = unix_timestamp();
        Self {
            content: split_lines(text),
            created_at: timestamp.clone(),
            modified_at: timestamp,
        }
    }

    pub fn text(&self) -> String {
        self.content.join("\n")
    }

    pub fn lines(&self) -> &[String] {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StateClaudeFileMiddleware {
    tool: AnthropicBuiltInTool,
    system_prompt: Option<&'static str>,
    allowed_path_prefixes: Vec<String>,
}

impl StateClaudeFileMiddleware {
    fn new(
        tool_kind: &'static str,
        tool_name: &'static str,
        system_prompt: Option<&'static str>,
    ) -> Self {
        Self {
            tool: AnthropicBuiltInTool::new(tool_kind, tool_name),
            system_prompt,
            allowed_path_prefixes: Vec::new(),
        }
    }

    fn with_allowed_path_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.allowed_path_prefixes
            .push(normalize_virtual_path(&prefix.into(), &[]).unwrap_or_else(|_| "/".to_owned()));
        self
    }

    fn tool(&self) -> AnthropicBuiltInTool {
        self.tool.clone()
    }

    fn system_prompt(&self) -> Option<&'static str> {
        self.system_prompt
    }

    fn view(
        &self,
        files: &BTreeMap<String, FileData>,
        path: &str,
        view_range: Option<(usize, usize)>,
    ) -> Result<String, LangChainError> {
        let normalized = normalize_virtual_path(path, &self.allowed_path_prefixes)?;
        if let Some(file) = files.get(&normalized) {
            return render_view(file.lines(), view_range);
        }

        let children = list_directory(files.keys(), &normalized);
        if children.is_empty() {
            return Err(LangChainError::request(format!(
                "virtual file `{normalized}` does not exist"
            )));
        }

        Ok(children.join("\n"))
    }

    fn create(
        &self,
        files: &mut BTreeMap<String, FileData>,
        path: &str,
        file_text: &str,
    ) -> Result<(), LangChainError> {
        let normalized = normalize_virtual_path(path, &self.allowed_path_prefixes)?;
        if files.contains_key(&normalized) {
            return Err(LangChainError::request(format!(
                "virtual file `{normalized}` already exists"
            )));
        }
        files.insert(normalized, FileData::new(file_text));
        Ok(())
    }

    fn str_replace(
        &self,
        files: &mut BTreeMap<String, FileData>,
        path: &str,
        old_str: &str,
        new_str: &str,
    ) -> Result<(), LangChainError> {
        let normalized = normalize_virtual_path(path, &self.allowed_path_prefixes)?;
        let file = files.get_mut(&normalized).ok_or_else(|| {
            LangChainError::request(format!("virtual file `{normalized}` does not exist"))
        })?;
        let current = file.text();
        if !current.contains(old_str) {
            return Err(LangChainError::request(format!(
                "`{old_str}` was not found in `{normalized}`"
            )));
        }
        *file = FileData::new(&current.replacen(old_str, new_str, 1));
        Ok(())
    }

    fn insert(
        &self,
        files: &mut BTreeMap<String, FileData>,
        path: &str,
        insert_line: usize,
        new_str: &str,
    ) -> Result<(), LangChainError> {
        let normalized = normalize_virtual_path(path, &self.allowed_path_prefixes)?;
        let file = files.get_mut(&normalized).ok_or_else(|| {
            LangChainError::request(format!("virtual file `{normalized}` does not exist"))
        })?;
        let mut lines = file.lines().to_vec();
        let index = insert_line.min(lines.len());
        lines.insert(index, new_str.to_owned());
        *file = FileData::new(&lines.join("\n"));
        Ok(())
    }

    fn delete(
        &self,
        files: &mut BTreeMap<String, FileData>,
        path: &str,
    ) -> Result<(), LangChainError> {
        let normalized = normalize_virtual_path(path, &self.allowed_path_prefixes)?;
        if files.remove(&normalized).is_none() {
            return Err(LangChainError::request(format!(
                "virtual file `{normalized}` does not exist"
            )));
        }
        Ok(())
    }

    fn rename(
        &self,
        files: &mut BTreeMap<String, FileData>,
        path: &str,
        new_path: &str,
    ) -> Result<(), LangChainError> {
        let from = normalize_virtual_path(path, &self.allowed_path_prefixes)?;
        let to = normalize_virtual_path(new_path, &self.allowed_path_prefixes)?;
        let file = files.remove(&from).ok_or_else(|| {
            LangChainError::request(format!("virtual file `{from}` does not exist"))
        })?;
        files.insert(to, file);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateClaudeTextEditorMiddleware {
    inner: StateClaudeFileMiddleware,
}

impl StateClaudeTextEditorMiddleware {
    pub fn new() -> Self {
        Self {
            inner: StateClaudeFileMiddleware::new(
                TEXT_EDITOR_TOOL_TYPE,
                TEXT_EDITOR_TOOL_NAME,
                None,
            ),
        }
    }

    pub fn with_allowed_path_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.inner = self.inner.with_allowed_path_prefix(prefix);
        self
    }

    pub fn tool(&self) -> AnthropicBuiltInTool {
        self.inner.tool()
    }

    pub fn system_prompt(&self) -> Option<&'static str> {
        self.inner.system_prompt()
    }

    pub fn view(
        &self,
        files: &BTreeMap<String, FileData>,
        path: &str,
        view_range: Option<(usize, usize)>,
    ) -> Result<String, LangChainError> {
        self.inner.view(files, path, view_range)
    }

    pub fn create(
        &self,
        files: &mut BTreeMap<String, FileData>,
        path: &str,
        file_text: &str,
    ) -> Result<(), LangChainError> {
        self.inner.create(files, path, file_text)
    }

    pub fn str_replace(
        &self,
        files: &mut BTreeMap<String, FileData>,
        path: &str,
        old_str: &str,
        new_str: &str,
    ) -> Result<(), LangChainError> {
        self.inner.str_replace(files, path, old_str, new_str)
    }

    pub fn insert(
        &self,
        files: &mut BTreeMap<String, FileData>,
        path: &str,
        insert_line: usize,
        new_str: &str,
    ) -> Result<(), LangChainError> {
        self.inner.insert(files, path, insert_line, new_str)
    }

    pub fn delete(
        &self,
        files: &mut BTreeMap<String, FileData>,
        path: &str,
    ) -> Result<(), LangChainError> {
        self.inner.delete(files, path)
    }

    pub fn rename(
        &self,
        files: &mut BTreeMap<String, FileData>,
        path: &str,
        new_path: &str,
    ) -> Result<(), LangChainError> {
        self.inner.rename(files, path, new_path)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateClaudeMemoryMiddleware {
    inner: StateClaudeFileMiddleware,
}

impl StateClaudeMemoryMiddleware {
    pub fn new() -> Self {
        Self {
            inner: StateClaudeFileMiddleware::new(
                MEMORY_TOOL_TYPE,
                MEMORY_TOOL_NAME,
                Some(MEMORY_SYSTEM_PROMPT),
            ),
        }
    }

    pub fn with_allowed_path_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.inner = self.inner.with_allowed_path_prefix(prefix);
        self
    }

    pub fn tool(&self) -> AnthropicBuiltInTool {
        self.inner.tool()
    }

    pub fn system_prompt(&self) -> Option<&'static str> {
        self.inner.system_prompt()
    }

    pub fn view(
        &self,
        files: &BTreeMap<String, FileData>,
        path: &str,
        view_range: Option<(usize, usize)>,
    ) -> Result<String, LangChainError> {
        self.inner.view(files, path, view_range)
    }

    pub fn create(
        &self,
        files: &mut BTreeMap<String, FileData>,
        path: &str,
        file_text: &str,
    ) -> Result<(), LangChainError> {
        self.inner.create(files, path, file_text)
    }

    pub fn str_replace(
        &self,
        files: &mut BTreeMap<String, FileData>,
        path: &str,
        old_str: &str,
        new_str: &str,
    ) -> Result<(), LangChainError> {
        self.inner.str_replace(files, path, old_str, new_str)
    }

    pub fn insert(
        &self,
        files: &mut BTreeMap<String, FileData>,
        path: &str,
        insert_line: usize,
        new_str: &str,
    ) -> Result<(), LangChainError> {
        self.inner.insert(files, path, insert_line, new_str)
    }

    pub fn delete(
        &self,
        files: &mut BTreeMap<String, FileData>,
        path: &str,
    ) -> Result<(), LangChainError> {
        self.inner.delete(files, path)
    }

    pub fn rename(
        &self,
        files: &mut BTreeMap<String, FileData>,
        path: &str,
        new_path: &str,
    ) -> Result<(), LangChainError> {
        self.inner.rename(files, path, new_path)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FilesystemClaudeFileMiddleware {
    root: PathBuf,
    tool: AnthropicBuiltInTool,
    system_prompt: Option<&'static str>,
    allowed_path_prefixes: Vec<String>,
}

impl FilesystemClaudeFileMiddleware {
    fn new(
        root: impl Into<PathBuf>,
        tool_kind: &'static str,
        tool_name: &'static str,
        system_prompt: Option<&'static str>,
    ) -> Self {
        Self {
            root: root.into(),
            tool: AnthropicBuiltInTool::new(tool_kind, tool_name),
            system_prompt,
            allowed_path_prefixes: Vec::new(),
        }
    }

    fn with_allowed_path_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.allowed_path_prefixes
            .push(normalize_virtual_path(&prefix.into(), &[]).unwrap_or_else(|_| "/".to_owned()));
        self
    }

    fn tool(&self) -> AnthropicBuiltInTool {
        self.tool.clone()
    }

    fn system_prompt(&self) -> Option<&'static str> {
        self.system_prompt
    }

    fn view(
        &self,
        path: &str,
        view_range: Option<(usize, usize)>,
    ) -> Result<String, LangChainError> {
        let resolved = self.resolve(path)?;
        if resolved.is_file() {
            let contents = fs::read_to_string(&resolved).map_err(io_error)?;
            return render_view(&split_lines(&contents), view_range);
        }

        if resolved.is_dir() {
            let mut entries = fs::read_dir(&resolved)
                .map_err(io_error)?
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter_map(|path| path.strip_prefix(&self.root).ok().map(Path::to_path_buf))
                .map(|path| format!("/{}", path.display()))
                .collect::<Vec<_>>();
            entries.sort();
            return Ok(entries.join("\n"));
        }

        Err(LangChainError::request(format!(
            "filesystem path `{}` does not exist",
            resolved.display()
        )))
    }

    fn create(&self, path: &str, file_text: &str) -> Result<(), LangChainError> {
        let resolved = self.resolve(path)?;
        if resolved.exists() {
            return Err(LangChainError::request(format!(
                "filesystem path `{}` already exists",
                resolved.display()
            )));
        }
        if let Some(parent) = resolved.parent() {
            fs::create_dir_all(parent).map_err(io_error)?;
        }
        fs::write(resolved, file_text).map_err(io_error)
    }

    fn str_replace(&self, path: &str, old_str: &str, new_str: &str) -> Result<(), LangChainError> {
        let resolved = self.resolve(path)?;
        let contents = fs::read_to_string(&resolved).map_err(io_error)?;
        if !contents.contains(old_str) {
            return Err(LangChainError::request(format!(
                "`{old_str}` was not found in `{}`",
                resolved.display()
            )));
        }
        fs::write(&resolved, contents.replacen(old_str, new_str, 1)).map_err(io_error)
    }

    fn insert(&self, path: &str, insert_line: usize, new_str: &str) -> Result<(), LangChainError> {
        let resolved = self.resolve(path)?;
        let contents = fs::read_to_string(&resolved).map_err(io_error)?;
        let mut lines = split_lines(&contents);
        let index = insert_line.min(lines.len());
        lines.insert(index, new_str.to_owned());
        fs::write(&resolved, lines.join("\n")).map_err(io_error)
    }

    fn delete(&self, path: &str) -> Result<(), LangChainError> {
        let resolved = self.resolve(path)?;
        if resolved.is_dir() {
            fs::remove_dir_all(resolved).map_err(io_error)
        } else {
            fs::remove_file(resolved).map_err(io_error)
        }
    }

    fn rename(&self, path: &str, new_path: &str) -> Result<(), LangChainError> {
        let from = self.resolve(path)?;
        let to = self.resolve(new_path)?;
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent).map_err(io_error)?;
        }
        fs::rename(from, to).map_err(io_error)
    }

    fn resolve(&self, path: &str) -> Result<PathBuf, LangChainError> {
        let normalized = normalize_virtual_path(path, &self.allowed_path_prefixes)?;
        Ok(self.root.join(normalized.trim_start_matches('/')))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilesystemClaudeTextEditorMiddleware {
    inner: FilesystemClaudeFileMiddleware,
}

impl FilesystemClaudeTextEditorMiddleware {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            inner: FilesystemClaudeFileMiddleware::new(
                root,
                TEXT_EDITOR_TOOL_TYPE,
                TEXT_EDITOR_TOOL_NAME,
                None,
            ),
        }
    }

    pub fn with_allowed_path_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.inner = self.inner.with_allowed_path_prefix(prefix);
        self
    }

    pub fn tool(&self) -> AnthropicBuiltInTool {
        self.inner.tool()
    }

    pub fn system_prompt(&self) -> Option<&'static str> {
        self.inner.system_prompt()
    }

    pub fn view(
        &self,
        path: &str,
        view_range: Option<(usize, usize)>,
    ) -> Result<String, LangChainError> {
        self.inner.view(path, view_range)
    }

    pub fn create(&self, path: &str, file_text: &str) -> Result<(), LangChainError> {
        self.inner.create(path, file_text)
    }

    pub fn str_replace(
        &self,
        path: &str,
        old_str: &str,
        new_str: &str,
    ) -> Result<(), LangChainError> {
        self.inner.str_replace(path, old_str, new_str)
    }

    pub fn insert(
        &self,
        path: &str,
        insert_line: usize,
        new_str: &str,
    ) -> Result<(), LangChainError> {
        self.inner.insert(path, insert_line, new_str)
    }

    pub fn delete(&self, path: &str) -> Result<(), LangChainError> {
        self.inner.delete(path)
    }

    pub fn rename(&self, path: &str, new_path: &str) -> Result<(), LangChainError> {
        self.inner.rename(path, new_path)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilesystemClaudeMemoryMiddleware {
    inner: FilesystemClaudeFileMiddleware,
}

impl FilesystemClaudeMemoryMiddleware {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            inner: FilesystemClaudeFileMiddleware::new(
                root,
                MEMORY_TOOL_TYPE,
                MEMORY_TOOL_NAME,
                Some(MEMORY_SYSTEM_PROMPT),
            ),
        }
    }

    pub fn with_allowed_path_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.inner = self.inner.with_allowed_path_prefix(prefix);
        self
    }

    pub fn tool(&self) -> AnthropicBuiltInTool {
        self.inner.tool()
    }

    pub fn system_prompt(&self) -> Option<&'static str> {
        self.inner.system_prompt()
    }

    pub fn view(
        &self,
        path: &str,
        view_range: Option<(usize, usize)>,
    ) -> Result<String, LangChainError> {
        self.inner.view(path, view_range)
    }

    pub fn create(&self, path: &str, file_text: &str) -> Result<(), LangChainError> {
        self.inner.create(path, file_text)
    }

    pub fn str_replace(
        &self,
        path: &str,
        old_str: &str,
        new_str: &str,
    ) -> Result<(), LangChainError> {
        self.inner.str_replace(path, old_str, new_str)
    }

    pub fn insert(
        &self,
        path: &str,
        insert_line: usize,
        new_str: &str,
    ) -> Result<(), LangChainError> {
        self.inner.insert(path, insert_line, new_str)
    }

    pub fn delete(&self, path: &str) -> Result<(), LangChainError> {
        self.inner.delete(path)
    }

    pub fn rename(&self, path: &str, new_path: &str) -> Result<(), LangChainError> {
        self.inner.rename(path, new_path)
    }
}

pub(crate) fn normalize_virtual_path(
    path: &str,
    allowed_prefixes: &[String],
) -> Result<String, LangChainError> {
    let mut segments = Vec::new();
    for component in Path::new(path).components() {
        match component {
            Component::RootDir | Component::CurDir => {}
            Component::Normal(value) => segments.push(value.to_string_lossy().to_string()),
            Component::ParentDir | Component::Prefix(_) => {
                return Err(LangChainError::request(format!(
                    "path traversal is not allowed: {path}"
                )));
            }
        }
    }

    let normalized = if segments.is_empty() {
        "/".to_owned()
    } else {
        format!("/{}", segments.join("/"))
    };

    if !allowed_prefixes.is_empty()
        && !allowed_prefixes
            .iter()
            .any(|prefix| normalized == *prefix || normalized.starts_with(&format!("{prefix}/")))
    {
        return Err(LangChainError::request(format!(
            "path `{normalized}` must start with one of {:?}",
            allowed_prefixes
        )));
    }

    Ok(normalized)
}

pub(crate) fn split_lines(text: &str) -> Vec<String> {
    if text.is_empty() {
        Vec::new()
    } else {
        text.lines().map(str::to_owned).collect()
    }
}

fn render_view(
    lines: &[String],
    view_range: Option<(usize, usize)>,
) -> Result<String, LangChainError> {
    if let Some((start, end)) = view_range {
        if start == 0 || end < start {
            return Err(LangChainError::request(
                "view ranges must be 1-based and inclusive",
            ));
        }
        let start_index = start.saturating_sub(1).min(lines.len());
        let end_index = end.min(lines.len());
        return Ok(lines[start_index..end_index].join("\n"));
    }

    Ok(lines.join("\n"))
}

fn list_directory<'a>(files: impl Iterator<Item = &'a String>, directory: &str) -> Vec<String> {
    let prefix = if directory == "/" {
        "/".to_owned()
    } else {
        format!("{directory}/")
    };
    let mut children = files
        .filter_map(|path| {
            if !path.starts_with(&prefix) {
                return None;
            }
            let relative = &path[prefix.len()..];
            (!relative.contains('/')).then(|| path.clone())
        })
        .collect::<Vec<_>>();
    children.sort();
    children
}

fn io_error(error: std::io::Error) -> LangChainError {
    LangChainError::request(error.to_string())
}

fn unix_timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_secs()
        .to_string()
}
