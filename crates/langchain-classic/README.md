# langchain-classic

Classic compatibility crate for legacy chains, memory, utilities, and classic aliases.

## Upstream Mapping

This crate maps to `libs/langchain` in the reference monorepo.

## Installation

```bash
cargo add langchain-classic
```

## Quick Start

```rust
use langchain_classic::chat_models::ParrotChatModel;

let model = ParrotChatModel::new("classic-chat-model", 12);
assert_eq!(model.model_name(), "classic-chat-model");
```

## Public Surface

- Legacy-facing namespaces such as `chains`, `memory`, `utilities`, `text_splitter`, `sql_database`
- Re-exported chat model, retriever, prompt, load, storage, and schema aliases
- `hub` prompt push/pull helpers for `PromptTemplate`, with overridable API URL/API key options
- Compatibility entry points for classic applications migrating from older layouts

## Hub Prompt Push/Pull

```rust
use langchain_classic::PromptTemplate;
use langchain_classic::hub::{self, HubOptions};

let options = HubOptions::new().with_api_url("http://127.0.0.1:8000");
let prompt = PromptTemplate::new("Hello {name}");

let _url = hub::push_with_options("owner/greeting", &prompt, &options)?;
let pulled = hub::pull_with_options("owner/greeting:latest", &options)?;
# Ok::<(), langchain_classic::LangChainError>(())
```

Current scope is intentionally narrow: only `PromptTemplate` is supported for hub transport in the Rust classic crate.

## Tests

- `tests/legacy_surface.rs`
- `tests/facade.rs`
- `tests/memory_extended.rs`
- `tests/memory_prompts.rs`
- `tests/utilities.rs`
