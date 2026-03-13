use std::process::Command;

use langchain_tests::integration_tests::{SandboxIntegrationHarness, SandboxIntegrationTests};
use langchain_tests::{
    BaseStandardTests, PYDANTIC_MAJOR_VERSION, StandardTestSuite, get_pydantic_major_version,
};

#[derive(Clone, Copy, Default)]
struct DefaultSandboxHarness;

impl SandboxIntegrationHarness for DefaultSandboxHarness {
    type Sandbox = ();

    fn sandbox(&self) -> Self::Sandbox {}
}

#[derive(Clone, Copy, Default)]
struct AsyncOnlySandboxHarness;

impl SandboxIntegrationHarness for AsyncOnlySandboxHarness {
    type Sandbox = ();

    fn sandbox(&self) -> Self::Sandbox {}

    fn has_sync(&self) -> bool {
        false
    }
}

fn probe_pydantic_major_version() -> usize {
    for interpreter in ["python3", "python"] {
        let output = Command::new(interpreter)
            .args([
                "-c",
                "import pydantic; print(int(pydantic.__version__.split('.')[0]))",
            ])
            .output();

        let Ok(output) = output else {
            continue;
        };

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Ok(value) = stdout.trim().parse::<usize>() {
                return value;
            }
        }
    }

    0
}

#[test]
fn base_standard_tests_helper_tracks_suite_name() {
    let helper = BaseStandardTests::new("chat_models");

    assert_eq!(helper.suite_name(), "chat_models");
}

#[derive(Clone, Copy)]
struct CleanSuite;

impl StandardTestSuite for CleanSuite {
    fn running_tests(&self) -> &'static [&'static str] {
        &["test_a", "test_b"]
    }

    fn base_tests(&self) -> &'static [&'static str] {
        &["test_a", "test_b"]
    }
}

#[derive(Clone, Copy)]
struct DriftedSuite;

impl StandardTestSuite for DriftedSuite {
    fn running_tests(&self) -> &'static [&'static str] {
        &["test_a", "test_override"]
    }

    fn base_tests(&self) -> &'static [&'static str] {
        &["test_a", "test_b", "test_override"]
    }

    fn overridden_tests(&self) -> &'static [&'static str] {
        &["test_override"]
    }
}

#[test]
fn base_standard_tests_detect_deleted_or_unmarked_overrides() {
    let helper = BaseStandardTests::new("chat_models");

    assert!(helper.assert_no_overrides(&CleanSuite).is_ok());

    let error = helper
        .assert_no_overrides(&DriftedSuite)
        .expect_err("drifted suite should be rejected");
    assert!(error.contains("Standard tests deleted"));
    assert!(error.contains("Standard tests overridden without xfail"));
}

#[test]
fn sandbox_integration_tests_exports_default_and_override_support_flags() {
    let default_suite = SandboxIntegrationTests::new(DefaultSandboxHarness);
    let async_only_suite = SandboxIntegrationTests::new(AsyncOnlySandboxHarness);

    assert_eq!(default_suite.base().suite_name(), "sandbox");
    assert!(default_suite.supports_sync());
    assert!(default_suite.supports_async());

    assert_eq!(async_only_suite.base().suite_name(), "sandbox");
    assert!(!async_only_suite.supports_sync());
    assert!(async_only_suite.supports_async());
}

#[test]
fn pydantic_major_version_is_discovered_dynamically() {
    let expected = probe_pydantic_major_version();

    assert_eq!(get_pydantic_major_version(), expected);
    assert_eq!(*PYDANTIC_MAJOR_VERSION, expected);
}
