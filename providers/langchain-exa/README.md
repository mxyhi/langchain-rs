# langchain-exa

Exa retriever and search-tool integration for web search and similar-result lookups.

## Upstream Mapping

This crate maps to `libs/partners/exa` in the reference monorepo.

## Installation

```bash
cargo add langchain-exa
```

## Quick Start

```rust
use langchain_exa::{ExaSearchRetriever, SearchHit};

let retriever = ExaSearchRetriever::new().with_hit(SearchHit::new(
    "LangChain",
    "https://example.com/langchain",
    "Build retrieval pipelines in Rust",
    0.0,
));
assert_eq!(retriever.max_results(), None);
```

## Public Surface

- `ExaSearchRetriever`
- `ExaSearchResults`, `ExaFindSimilarResults`, `SearchHit`
- `TextContentsOptions`, `HighlightsContentsOptions`
- `retrievers`, `tools`, `types` namespaces

## Tests

- `tests/retriever_and_tools.rs`
- `tests/namespace.rs`

