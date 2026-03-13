use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{LazyLock, Mutex};

use langchain_model_profiles::cli::{
    describe_provider, render_capability_table, render_provider_detail, render_provider_table, run,
};
use langchain_model_profiles::provider;
use serde_json::Value;
use tempfile::TempDir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

static CATALOG_ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
const CATALOG_URL_ENV: &str = "LANGCHAIN_MODEL_PROFILES_CATALOG_URL";

struct CatalogUrlEnvGuard {
    previous: Option<std::ffi::OsString>,
}

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

fn with_catalog_url_env<T>(value: &str, run_test: impl FnOnce() -> T) -> T {
    let _guard = CATALOG_ENV_LOCK
        .lock()
        .expect("env lock should not be poisoned");
    let _env_guard = CatalogUrlEnvGuard::set(value);
    let result = run_test();
    result
}

impl CatalogUrlEnvGuard {
    fn set(value: &str) -> Self {
        let previous = std::env::var_os(CATALOG_URL_ENV);
        unsafe {
            std::env::set_var(CATALOG_URL_ENV, value);
        }
        Self { previous }
    }
}

impl Drop for CatalogUrlEnvGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(previous) => unsafe {
                std::env::set_var(CATALOG_URL_ENV, previous);
            },
            None => unsafe {
                std::env::remove_var(CATALOG_URL_ENV);
            },
        }
    }
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
fn provider_exports_match_huggingface_and_exa_public_surface() {
    let huggingface = provider("huggingface").expect("huggingface profile should exist");
    assert!(
        huggingface.exports.contains(&"HuggingFacePipeline"),
        "huggingface exports should keep HuggingFacePipeline in sync with provider surface"
    );

    let exa = provider("exa").expect("exa profile should exist");
    assert!(
        exa.exports.contains(&"HighlightsContentsOptions"),
        "exa exports should keep HighlightsContentsOptions in sync with provider surface"
    );
    assert!(
        exa.exports.contains(&"TextContentsOptions"),
        "exa exports should keep TextContentsOptions in sync with provider surface"
    );
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

#[test]
fn cli_run_refresh_without_catalog_uses_live_catalog_url_override() {
    let runtime = tokio::runtime::Runtime::new().expect("runtime should start");
    let server = runtime.block_on(async {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api.json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "anthropic": {
                    "models": {
                        "claude-live-contract": {
                            "limit": {
                                "context": 4096,
                                "output": 2048
                            },
                            "modalities": {
                                "input": ["text"],
                                "output": ["text"]
                            }
                        }
                    }
                }
            })))
            .mount(&server)
            .await;
        server
    });

    let (_tempdir, data_dir) = temp_data_dir();
    write_augmentations(&data_dir);

    let result = with_catalog_url_env(&format!("{}/api.json", server.uri()), || {
        run([
            "refresh",
            "--provider",
            "anthropic",
            "--data-dir",
            data_dir.to_str().expect("temp path should be utf8"),
        ])
    })
    .expect("refresh run should use overridden live catalog");

    assert!(result.contains("provider: anthropic"));
    let generated = read_profiles_json(&data_dir);
    assert!(generated["models"].get("claude-live-contract").is_some());
}

#[test]
fn cli_run_refresh_without_catalog_surfaces_live_fetch_failures() {
    let runtime = tokio::runtime::Runtime::new().expect("runtime should start");
    let server = runtime.block_on(async {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api.json"))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream unavailable"))
            .mount(&server)
            .await;
        server
    });

    let (_tempdir, data_dir) = temp_data_dir();

    let error = with_catalog_url_env(&format!("{}/api.json", server.uri()), || {
        run([
            "refresh",
            "--provider",
            "anthropic",
            "--data-dir",
            data_dir.to_str().expect("temp path should be utf8"),
        ])
    })
    .expect_err("refresh should report live catalog failures");

    assert!(
        error
            .to_string()
            .contains(&format!("{}/api.json", server.uri()))
    );
    assert!(error.to_string().contains("request failed"));
}
