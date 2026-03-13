# langchain-nomic

Nomic embeddings integration for remote embedding APIs.

## Upstream Mapping

This crate maps to `libs/partners/nomic` in the reference monorepo.

## Installation

```bash
cargo add langchain-nomic
```

## Quick Start

```rust
use langchain_nomic::NomicEmbeddings;

let embeddings = NomicEmbeddings::new("nomic-embed-text-v1.5");
assert_eq!(embeddings.model(), "nomic-embed-text-v1.5");
```

## Public Surface

- `NomicEmbeddings`
- `embeddings::NomicEmbeddings`
- Local-default and remote-base-url embedding constructors

## Tests

- `tests/boundaries.rs`
- `tests/namespace.rs`

