# langchain-chroma

Chroma vectorstore integration for local or remote collection-backed retrieval.

## Upstream Mapping

This crate maps to `libs/partners/chroma` in the reference monorepo.

## Installation

```bash
cargo add langchain-chroma
```

## Quick Start

```rust
use langchain_chroma::vectorstores::Chroma;
use langchain_core::embeddings::CharacterEmbeddings;

let store = Chroma::new("docs", CharacterEmbeddings::new());
assert_eq!(store.collection_name(), "docs");
```

## Public Surface

- `vectorstores::Chroma`
- Local and remote constructors including `new_remote` and `new_remote_with_namespace`
- Collection-focused vectorstore operations

## Tests

- `tests/namespace.rs`
- `tests/vectorstore.rs`

