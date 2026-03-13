use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root should exist")
        .to_path_buf()
}

fn read_file(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn sorted_child_dirs(path: &Path) -> Vec<String> {
    let mut entries = fs::read_dir(path)
        .unwrap_or_else(|error| panic!("failed to read directory {}: {error}", path.display()))
        .filter_map(Result::ok)
        .filter_map(|entry| {
            entry
                .file_type()
                .ok()
                .filter(|file_type| file_type.is_dir())
                .and_then(|_| entry.file_name().into_string().ok())
        })
        .collect::<Vec<_>>();
    entries.sort();
    entries
}

#[test]
fn workspace_inventory_matches_reference_monorepo_packages() {
    let root = workspace_root();
    let reference_packages = sorted_child_dirs(&root.join(".ref/langchain/libs"));
    let expected_reference = vec![
        "core",
        "langchain",
        "langchain_v1",
        "model-profiles",
        "partners",
        "standard-tests",
        "text-splitters",
    ];
    assert_eq!(reference_packages, expected_reference);

    for (reference, rust_path) in [
        ("core", "crates/langchain-core"),
        ("langchain", "crates/langchain-classic"),
        ("langchain_v1", "crates/langchain"),
        ("model-profiles", "crates/langchain-model-profiles"),
        ("partners", "providers"),
        ("standard-tests", "crates/langchain-tests"),
        ("text-splitters", "crates/langchain-text-splitters"),
    ] {
        assert!(
            root.join(".ref/langchain/libs").join(reference).exists(),
            "reference package {reference} should exist"
        );
        assert!(
            root.join(rust_path).exists(),
            "workspace path {rust_path} should exist for reference package {reference}"
        );
    }
}

#[test]
fn provider_directories_match_reference_partner_set() {
    let root = workspace_root();
    let reference_partners = sorted_child_dirs(&root.join(".ref/langchain/libs/partners"));
    let provider_dirs = sorted_child_dirs(&root.join("providers"))
        .into_iter()
        .map(|directory| {
            directory
                .strip_prefix("langchain-")
                .unwrap_or_else(|| {
                    panic!("provider directory {directory} should start with langchain-")
                })
                .to_owned()
        })
        .collect::<Vec<_>>();

    assert_eq!(provider_dirs, reference_partners);
}

#[test]
fn every_workspace_package_has_readme_with_upstream_mapping() {
    let root = workspace_root();
    let package_dirs = sorted_child_dirs(&root.join("crates"))
        .into_iter()
        .map(|directory| format!("crates/{directory}/README.md"))
        .chain(
            sorted_child_dirs(&root.join("providers"))
                .into_iter()
                .map(|directory| format!("providers/{directory}/README.md")),
        )
        .collect::<Vec<_>>();

    for relative_path in package_dirs {
        let readme_path = root.join(&relative_path);
        let readme = read_file(&readme_path);

        for required in [
            "## Upstream Mapping",
            "## Quick Start",
            "## Public Surface",
            "## Tests",
        ] {
            assert!(
                readme.contains(required),
                "{} should contain {required:?}",
                readme_path.display()
            );
        }
    }
}

#[test]
fn root_readme_documents_reference_parity_contract() {
    let readme = read_file(&workspace_root().join("README.md"));

    for required in [
        "## Reference Parity",
        "`libs/partners/*` -> `providers/langchain-*`",
        "`cargo test -p langchain --test reference_inventory_parity`",
    ] {
        assert!(
            readme.contains(required),
            "root README should contain {required:?}"
        );
    }
}
