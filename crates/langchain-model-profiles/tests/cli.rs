use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use langchain_model_profiles::cli::{
    describe_provider, render_capability_table, render_provider_detail, render_provider_table, run,
};
use serde_json::Value;
use tempfile::TempDir;

fn langchain_profiles() -> Command {
    let path =
        std::env::var("CARGO_BIN_EXE_langchain-profiles").expect("binary path should be set");
    Command::new(path)
}

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn temp_data_dir() -> (TempDir, PathBuf) {
    let tempdir = tempfile::tempdir().expect("temp dir should be created");
    let data_dir = tempdir.path().join("data");
    (tempdir, data_dir)
}

fn write_augmentations(data_dir: &Path) {
    fs::create_dir_all(data_dir).expect("data dir should be created");
    fs::write(
        data_dir.join("profile_augmentations.toml"),
        r#"
[overrides]
image_url_inputs = true
pdf_inputs = true

[overrides."claude-3-opus"]
max_output_tokens = 8192

[overrides."custom-offline-model"]
structured_output = true
max_input_tokens = 123
"#,
    )
    .expect("augmentations should be written");
}

fn read_profiles_json(data_dir: &Path) -> Value {
    let bytes = fs::read(data_dir.join("_profiles.json")).expect("profiles output should exist");
    serde_json::from_slice(&bytes).expect("profiles output should be valid json")
}

#[test]
fn cli_helpers_describe_known_provider_and_render_table() {
    let openai = describe_provider("openai").expect("openai profile should exist");
    assert_eq!(openai.package_name, "langchain-openai");
    assert!(openai.supports_chat_model());
    assert!(openai.supports_embeddings());

    let table = render_provider_table();
    assert!(table.contains("provider"));
    assert!(table.contains("openai"));
    assert!(table.contains("langchain-openai"));
    assert!(table.contains("anthropic"));
}

#[test]
fn cli_helpers_render_provider_detail_and_capability_views() {
    let detail = render_provider_detail("perplexity").expect("perplexity should exist");
    assert!(detail.contains("provider: perplexity"));
    assert!(detail.contains("package: langchain-perplexity"));
    assert!(detail.contains("retriever: yes"));
    assert!(detail.contains("parser_or_tooling: yes"));

    let embeddings = render_capability_table("embeddings").expect("capability should exist");
    assert!(embeddings.contains("capability: embeddings"));
    assert!(embeddings.contains("openai"));
    assert!(embeddings.contains("mistralai"));
    assert!(embeddings.contains("nomic"));
    assert!(!embeddings.contains("anthropic"));
}

#[test]
fn binary_list_subcommand_prints_provider_table() {
    let output = langchain_profiles()
        .arg("list")
        .output()
        .expect("langchain-profiles binary should run");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf8");
    assert!(stdout.contains("provider"));
    assert!(stdout.contains("openai"));
    assert!(stdout.contains("langchain-openai"));
}

#[test]
fn binary_show_and_provider_subcommands_render_provider_detail() {
    let show = langchain_profiles()
        .args(["show", "openai"])
        .output()
        .expect("show subcommand should run");
    assert!(show.status.success());
    let show_stdout = String::from_utf8(show.stdout).expect("stdout should be utf8");
    assert!(show_stdout.contains("provider: openai"));
    assert!(show_stdout.contains("default_base_url: https://api.openai.com/v1"));

    let provider = langchain_profiles()
        .args(["provider", "perplexity"])
        .output()
        .expect("provider subcommand should run");
    assert!(provider.status.success());
    let provider_stdout = String::from_utf8(provider.stdout).expect("stdout should be utf8");
    assert!(provider_stdout.contains("provider: perplexity"));
    assert!(provider_stdout.contains("retriever: yes"));
}

#[test]
fn binary_capability_subcommand_lists_supporting_providers() {
    let output = langchain_profiles()
        .args(["capability", "chat_model"])
        .output()
        .expect("capability subcommand should run");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf8");
    assert!(stdout.contains("capability: chat_model"));
    assert!(stdout.contains("openai"));
    assert!(stdout.contains("anthropic"));
    assert!(stdout.contains("huggingface"));
}

#[test]
fn binary_unknown_provider_fails_with_stderr_message() {
    let output = langchain_profiles()
        .args(["show", "does-not-exist"])
        .output()
        .expect("show subcommand should run");

    assert!(!output.status.success());

    let stderr = String::from_utf8(output.stderr).expect("stderr should be utf8");
    assert!(stderr.contains("unknown provider"));
    assert!(stderr.contains("does-not-exist"));
}

#[test]
fn cli_run_refresh_generates_profiles_json_from_catalog_fixture() {
    let (_tempdir, data_dir) = temp_data_dir();
    write_augmentations(&data_dir);

    let output = run([
        "refresh",
        "--provider",
        "anthropic",
        "--data-dir",
        data_dir.to_str().expect("temp path should be utf8"),
        "--catalog",
        fixture_path("models-dev-sample.json")
            .to_str()
            .expect("fixture path should be utf8"),
    ])
    .expect("refresh run should succeed");

    assert!(output.contains("provider: anthropic"));
    assert!(output.contains("_profiles.json"));

    let generated = read_profiles_json(&data_dir);
    assert_eq!(generated["provider"], "anthropic");
    assert_eq!(generated["package_name"], "langchain-anthropic");
    assert_eq!(generated["default_base_url"], "https://api.anthropic.com");
    assert_eq!(
        generated["models"]["claude-3-opus"]["max_output_tokens"],
        8192
    );
    assert_eq!(
        generated["models"]["claude-3-opus"]["image_url_inputs"],
        true
    );
    assert_eq!(generated["models"]["claude-3-opus"]["pdf_inputs"], true);
    assert_eq!(
        generated["models"]["custom-offline-model"]["max_input_tokens"],
        123
    );
    assert_eq!(
        generated["models"]["custom-offline-model"]["structured_output"],
        true
    );
}

#[test]
fn binary_refresh_subcommand_writes_profiles_json() {
    let (_tempdir, data_dir) = temp_data_dir();
    write_augmentations(&data_dir);

    let output = langchain_profiles()
        .args([
            "refresh",
            "--provider",
            "anthropic",
            "--data-dir",
            data_dir.to_str().expect("temp path should be utf8"),
            "--catalog",
            fixture_path("models-dev-sample.json")
                .to_str()
                .expect("fixture path should be utf8"),
        ])
        .output()
        .expect("refresh subcommand should run");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf8");
    assert!(stdout.contains("provider: anthropic"));
    assert!(stdout.contains("models: 3"));
    assert!(data_dir.join("_profiles.json").exists());
}

#[test]
fn binary_refresh_subcommand_fails_for_unknown_provider() {
    let (_tempdir, data_dir) = temp_data_dir();

    let output = langchain_profiles()
        .args([
            "refresh",
            "--provider",
            "does-not-exist",
            "--data-dir",
            data_dir.to_str().expect("temp path should be utf8"),
            "--catalog",
            fixture_path("models-dev-sample.json")
                .to_str()
                .expect("fixture path should be utf8"),
        ])
        .output()
        .expect("refresh subcommand should run");

    assert!(!output.status.success());

    let stderr = String::from_utf8(output.stderr).expect("stderr should be utf8");
    assert!(stderr.contains("provider not found"));
    assert!(stderr.contains("does-not-exist"));
}
