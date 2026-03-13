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
- Compatibility entry points for classic applications migrating from older layouts

## Tests

- `tests/legacy_surface.rs`
- `tests/facade.rs`
- `tests/memory_extended.rs`
- `tests/memory_prompts.rs`

