use std::fs;
use std::path::{Path, PathBuf};

struct ReadmeExpectation {
    rel_path: &'static str,
    heading: &'static str,
    upstream: &'static str,
    install: &'static str,
}

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

#[test]
fn root_readme_describes_workspace_mapping_and_usage() {
    let readme = read_file(&workspace_root().join("README.md"));

    for required in [
        "# langchain-rs",
        ".ref/langchain",
        "## Workspace Layout",
        "`crates/langchain` = `libs/langchain_v1`",
        "`crates/langchain-classic` = `libs/langchain`",
        "## Quick Start",
        "## Provider Matrix",
        "cargo test --workspace",
        "langchain-openai",
        "langchain-text-splitters",
    ] {
        assert!(
            readme.contains(required),
            "root README should contain {required:?}"
        );
    }
}

#[test]
fn package_readmes_exist_and_include_core_sections() {
    let expectations = [
        ReadmeExpectation {
            rel_path: "crates/langchain-core/README.md",
            heading: "# langchain-core",
            upstream: "libs/core",
            install: "cargo add langchain-core",
        },
        ReadmeExpectation {
            rel_path: "crates/langchain/README.md",
            heading: "# langchain",
            upstream: "libs/langchain_v1",
            install: "cargo add langchain",
        },
        ReadmeExpectation {
            rel_path: "crates/langchain-classic/README.md",
            heading: "# langchain-classic",
            upstream: "libs/langchain",
            install: "cargo add langchain-classic",
        },
        ReadmeExpectation {
            rel_path: "crates/langchain-model-profiles/README.md",
            heading: "# langchain-model-profiles",
            upstream: "libs/model-profiles",
            install: "cargo add langchain-model-profiles",
        },
        ReadmeExpectation {
            rel_path: "crates/langchain-tests/README.md",
            heading: "# langchain-tests",
            upstream: "libs/standard-tests",
            install: "cargo add langchain-tests --dev",
        },
        ReadmeExpectation {
            rel_path: "crates/langchain-text-splitters/README.md",
            heading: "# langchain-text-splitters",
            upstream: "libs/text-splitters",
            install: "cargo add langchain-text-splitters",
        },
        ReadmeExpectation {
            rel_path: "providers/langchain-openai/README.md",
            heading: "# langchain-openai",
            upstream: "libs/partners/openai",
            install: "cargo add langchain-openai",
        },
        ReadmeExpectation {
            rel_path: "providers/langchain-anthropic/README.md",
            heading: "# langchain-anthropic",
            upstream: "libs/partners/anthropic",
            install: "cargo add langchain-anthropic",
        },
        ReadmeExpectation {
            rel_path: "providers/langchain-chroma/README.md",
            heading: "# langchain-chroma",
            upstream: "libs/partners/chroma",
            install: "cargo add langchain-chroma",
        },
        ReadmeExpectation {
            rel_path: "providers/langchain-deepseek/README.md",
            heading: "# langchain-deepseek",
            upstream: "libs/partners/deepseek",
            install: "cargo add langchain-deepseek",
        },
        ReadmeExpectation {
            rel_path: "providers/langchain-exa/README.md",
            heading: "# langchain-exa",
            upstream: "libs/partners/exa",
            install: "cargo add langchain-exa",
        },
        ReadmeExpectation {
            rel_path: "providers/langchain-fireworks/README.md",
            heading: "# langchain-fireworks",
            upstream: "libs/partners/fireworks",
            install: "cargo add langchain-fireworks",
        },
        ReadmeExpectation {
            rel_path: "providers/langchain-groq/README.md",
            heading: "# langchain-groq",
            upstream: "libs/partners/groq",
            install: "cargo add langchain-groq",
        },
        ReadmeExpectation {
            rel_path: "providers/langchain-huggingface/README.md",
            heading: "# langchain-huggingface",
            upstream: "libs/partners/huggingface",
            install: "cargo add langchain-huggingface",
        },
        ReadmeExpectation {
            rel_path: "providers/langchain-mistralai/README.md",
            heading: "# langchain-mistralai",
            upstream: "libs/partners/mistralai",
            install: "cargo add langchain-mistralai",
        },
        ReadmeExpectation {
            rel_path: "providers/langchain-nomic/README.md",
            heading: "# langchain-nomic",
            upstream: "libs/partners/nomic",
            install: "cargo add langchain-nomic",
        },
        ReadmeExpectation {
            rel_path: "providers/langchain-ollama/README.md",
            heading: "# langchain-ollama",
            upstream: "libs/partners/ollama",
            install: "cargo add langchain-ollama",
        },
        ReadmeExpectation {
            rel_path: "providers/langchain-openrouter/README.md",
            heading: "# langchain-openrouter",
            upstream: "libs/partners/openrouter",
            install: "cargo add langchain-openrouter",
        },
        ReadmeExpectation {
            rel_path: "providers/langchain-perplexity/README.md",
            heading: "# langchain-perplexity",
            upstream: "libs/partners/perplexity",
            install: "cargo add langchain-perplexity",
        },
        ReadmeExpectation {
            rel_path: "providers/langchain-qdrant/README.md",
            heading: "# langchain-qdrant",
            upstream: "libs/partners/qdrant",
            install: "cargo add langchain-qdrant",
        },
        ReadmeExpectation {
            rel_path: "providers/langchain-xai/README.md",
            heading: "# langchain-xai",
            upstream: "libs/partners/xai",
            install: "cargo add langchain-xai",
        },
    ];

    for expectation in expectations {
        let readme_path = workspace_root().join(expectation.rel_path);
        assert!(
            readme_path.exists(),
            "README should exist at {}",
            readme_path.display()
        );

        let readme = read_file(&readme_path);
        for required in [
            expectation.heading,
            "## Upstream Mapping",
            expectation.upstream,
            "## Installation",
            expectation.install,
            "## Quick Start",
            "## Public Surface",
            "## Tests",
        ] {
            assert!(
                readme.contains(required),
                "{} should contain {:?}",
                readme_path.display(),
                required
            );
        }
    }
}

#[test]
fn provider_readmes_document_public_data_namespaces_when_exposed() {
    let expectations = [
        (
            "providers/langchain-anthropic/README.md",
            "data::anthropic_profile()",
        ),
        (
            "providers/langchain-deepseek/README.md",
            "data::deepseek_profile()",
        ),
        (
            "providers/langchain-fireworks/README.md",
            "data::fireworks_profile()",
        ),
        ("providers/langchain-groq/README.md", "data::groq_profile()"),
        (
            "providers/langchain-mistralai/README.md",
            "data::mistralai_profile()",
        ),
        (
            "providers/langchain-ollama/README.md",
            "data::ollama_profile()",
        ),
        (
            "providers/langchain-openrouter/README.md",
            "data::openrouter_profile()",
        ),
        (
            "providers/langchain-openai/README.md",
            "data::openai_profile()",
        ),
        (
            "providers/langchain-huggingface/README.md",
            "data::huggingface_profile()",
        ),
        (
            "providers/langchain-perplexity/README.md",
            "data::perplexity_profile()",
        ),
        ("providers/langchain-xai/README.md", "data::xai_profile()"),
    ];

    for (rel_path, required) in expectations {
        let readme_path = workspace_root().join(rel_path);
        let readme = read_file(&readme_path);
        assert!(
            readme.contains(required),
            "{} should document its public data namespace via {:?}",
            readme_path.display(),
            required
        );
    }
}

#[test]
fn exa_readme_tracks_the_actual_root_api_surface() {
    let readme_path = workspace_root().join("providers/langchain-exa/README.md");
    let readme = read_file(&readme_path);

    for required in [
        "ExaSearchResults::new()",
        "ExaFindSimilarResults::new()",
        "TextContentsOptions::default().with_max_characters(",
    ] {
        assert!(
            readme.contains(required),
            "{} should document exa root API via {:?}",
            readme_path.display(),
            required
        );
    }

    assert!(
        !readme.contains("max_results()"),
        "{} should not document nonexistent ExaSearchRetriever::max_results()",
        readme_path.display()
    );
}

#[test]
fn perplexity_readme_matches_current_api_surface() {
    let readme_path = workspace_root().join("providers/langchain-perplexity/README.md");
    let readme = read_file(&readme_path);

    for required in [
        "use langchain_core::language_models::BaseChatModel;",
        "let model = ChatPerplexity::new(\"sonar\")",
        "`UserLocation`, `MediaResponse`, `MediaResponseOverrides`, `WebSearchOptions`",
    ] {
        assert!(
            readme.contains(required),
            "{} should contain {:?}",
            readme_path.display(),
            required
        );
    }

    assert!(
        !readme.contains("ChatPerplexity::new(\"sonar\", None::<&str>)"),
        "{} should not describe the removed two-argument constructor",
        readme_path.display()
    );
}
