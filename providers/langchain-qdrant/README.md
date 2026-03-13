# langchain-qdrant

Qdrant vectorstore integration with dense, sparse, and hybrid retrieval modes.

## Upstream Mapping

This crate maps to `libs/partners/qdrant` in the reference monorepo.

## Installation

```bash
cargo add langchain-qdrant
```

## Quick Start

```rust
use langchain_core::embeddings::CharacterEmbeddings;
use langchain_qdrant::{Qdrant, RetrievalMode};

let store = Qdrant::new(CharacterEmbeddings::new()).with_retrieval_mode(RetrievalMode::Hybrid);
assert_eq!(store.retrieval_mode(), RetrievalMode::Hybrid);
```

## Public Surface

- `QdrantVectorStore`, `Qdrant`
- `RetrievalMode`, `SparseEmbeddings`, `SparseVector`, `FastEmbedSparse`
- `vectorstores`, `qdrant`, `fastembed_sparse`, `sparse_embeddings` namespaces

## Tests

- `tests/namespace.rs`
- `tests/vectorstore.rs`

