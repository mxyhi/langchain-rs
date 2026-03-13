use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use langchain_anthropic::middleware::{
    ClaudeBashToolMiddleware, FilesystemClaudeMemoryMiddleware,
    FilesystemClaudeTextEditorMiddleware, GrepOutputMode, StateClaudeMemoryMiddleware,
    StateClaudeTextEditorMiddleware, StateFileSearchMiddleware,
};
use serde_json::json;

fn temp_path(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("langchain-anthropic-{label}-{nanos}"))
}

#[test]
fn state_text_editor_and_memory_middlewares_apply_file_operations() {
    let editor = StateClaudeTextEditorMiddleware::new();
    let mut files = BTreeMap::new();

    editor
        .create(&mut files, "/notes/todo.md", "line one\nline two")
        .expect("create should succeed");
    assert_eq!(
        editor
            .view(&files, "/notes/todo.md", None)
            .expect("view should succeed"),
        "line one\nline two"
    );

    editor
        .str_replace(&mut files, "/notes/todo.md", "line two", "line 2")
        .expect("replace should succeed");
    editor
        .insert(&mut files, "/notes/todo.md", 2, "line three")
        .expect("insert should succeed");
    assert_eq!(
        editor
            .view(&files, "/notes/todo.md", Some((2, 3)))
            .expect("range view should succeed"),
        "line 2\nline three"
    );

    let memory = StateClaudeMemoryMiddleware::new();
    memory
        .rename(&mut files, "/notes/todo.md", "/notes/archive.md")
        .expect("rename should succeed");
    assert!(files.contains_key("/notes/archive.md"));
    memory
        .delete(&mut files, "/notes/archive.md")
        .expect("delete should succeed");
    assert!(files.is_empty());
}

#[test]
fn filesystem_middlewares_roundtrip_files() {
    let root = temp_path("filesystem");
    fs::create_dir_all(&root).expect("temp root should be created");

    let editor = FilesystemClaudeTextEditorMiddleware::new(&root);
    editor
        .create("/project/notes.txt", "alpha\nbeta")
        .expect("filesystem create should succeed");
    editor
        .str_replace("/project/notes.txt", "beta", "bravo")
        .expect("filesystem replace should succeed");

    let memory = FilesystemClaudeMemoryMiddleware::new(&root);
    assert_eq!(
        memory
            .view("/project/notes.txt", None)
            .expect("filesystem view should succeed"),
        "alpha\nbravo"
    );

    memory
        .rename("/project/notes.txt", "/project/archive.txt")
        .expect("filesystem rename should succeed");
    assert!(
        root.join("project/archive.txt").exists(),
        "renamed file should exist"
    );

    memory
        .delete("/project/archive.txt")
        .expect("filesystem delete should succeed");
    assert!(
        !root.join("project/archive.txt").exists(),
        "deleted file should be removed"
    );

    fs::remove_dir_all(&root).expect("temp root should be removed");
}

#[test]
fn state_file_search_matches_glob_and_grep_queries() {
    let editor = StateClaudeTextEditorMiddleware::new();
    let search = StateFileSearchMiddleware::new();
    let mut files = BTreeMap::new();

    editor
        .create(
            &mut files,
            "/workspace/src/main.rs",
            "fn main() {\n    println!(\"hi\");\n}",
        )
        .expect("main file should be created");
    editor
        .create(
            &mut files,
            "/workspace/src/lib.rs",
            "pub fn greet() -> &'static str {\n    \"hello\"\n}",
        )
        .expect("lib file should be created");

    let glob_hits = search
        .glob_search("**/*.rs", "/workspace", &files)
        .expect("glob search should succeed");
    assert_eq!(glob_hits.len(), 2);
    assert!(glob_hits.iter().any(|path| path.ends_with("main.rs")));

    let grep_hits = search
        .grep_search(
            "println!",
            "/workspace",
            Some("*.rs"),
            GrepOutputMode::Content,
            &files,
        )
        .expect("grep search should succeed");
    assert_eq!(grep_hits.len(), 1);
    assert_eq!(grep_hits[0].path, "/workspace/src/main.rs");
    assert_eq!(grep_hits[0].line_number, Some(2));
    assert!(
        grep_hits[0]
            .line
            .as_deref()
            .expect("content mode should keep the line")
            .contains("println!")
    );
}

#[test]
fn anthropic_bash_and_tool_parser_helpers_are_executable() {
    let root = temp_path("bash");
    fs::create_dir_all(&root).expect("temp root should be created");

    let bash = ClaudeBashToolMiddleware::new(Some(&root));
    let output = bash
        .execute("printf 'anthropic'")
        .expect("bash should execute");
    assert_eq!(output.status_code(), 0);
    assert_eq!(output.stdout(), "anthropic");

    let parser = langchain_anthropic::output_parsers::ToolsOutputParser::new();
    let message = langchain_core::messages::AIMessage::new(String::new()).with_tool_calls(vec![
        langchain_core::messages::ToolCall::new(
            "lookup",
            json!({ "query": "rust", "tags": ["langchain", "anthropic"] }),
        )
        .with_id("toolu_123"),
    ]);
    let args = parser.parse_tool_args(&message);
    assert_eq!(
        args,
        vec![json!({ "query": "rust", "tags": ["langchain", "anthropic"] })]
    );

    let structured = parser
        .parse_structured::<LookupArgs>(&message)
        .expect("structured parsing should succeed");
    assert_eq!(
        structured,
        vec![LookupArgs {
            query: "rust".to_owned(),
            tags: vec!["langchain".to_owned(), "anthropic".to_owned()],
        }]
    );

    fs::remove_dir_all(&root).expect("temp root should be removed");
}

#[test]
fn experimental_tool_call_parser_understands_nested_and_repeated_parameters() {
    let xml = r#"
<function_calls>
  <invoke>
    <tool_name>lookup</tool_name>
    <parameters>
      <query>rust</query>
      <tags>langchain</tags>
      <tags>anthropic</tags>
      <filters>
        <status>stable</status>
      </filters>
    </parameters>
  </invoke>
</function_calls>
"#;

    let calls = langchain_anthropic::experimental::extract_tool_calls(
        xml,
        &[
            langchain_core::tools::tool("lookup", "Look up data").with_parameters(json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "tags": { "type": "array", "items": { "type": "string" } },
                    "filters": {
                        "type": "object",
                        "properties": {
                            "status": { "type": "string" }
                        }
                    }
                }
            })),
        ],
    )
    .expect("experimental parser should succeed");

    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].args(),
        &json!({
            "query": "rust",
            "tags": ["langchain", "anthropic"],
            "filters": { "status": "stable" }
        })
    );
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
struct LookupArgs {
    query: String,
    tags: Vec<String>,
}
