# langchain-model-profiles

Workspace-maintained provider profile registry and CLI for inspecting or refreshing model capability data.

## Upstream Mapping

This crate maps to `libs/model-profiles` in the reference monorepo.

## Installation

```bash
cargo add langchain-model-profiles
```

## Quick Start

```bash
langchain-profiles list
langchain-profiles show openai
langchain-profiles refresh --provider openai --data-dir ./providers/langchain-openai/data
langchain-profiles refresh --provider anthropic --data-dir ./tmp/data --catalog ./tests/fixtures/models-dev-sample.json
```

## Public Surface

- Provider registry lookup via `provider()` and `providers()`
- Capability metadata through `ProviderProfile` and `ProviderCapabilities`
- CLI subcommands for listing, filtering, inspecting, and refreshing profile data
- `refresh` merges `profile_augmentations.toml` and writes `_profiles.json`

## Tests

- `tests/cli.rs`
