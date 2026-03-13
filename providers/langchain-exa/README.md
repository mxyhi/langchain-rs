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
use langchain_exa::{
    ExaFindSimilarResults, ExaSearchResults, ExaSearchRetriever, SearchHit, TextContentsOptions,
};

let search = ExaSearchResults::new()
    .with_hit(SearchHit::new(
        "Rust async guide",
        "https://example.com/rust",
        "Rust async runtime and futures",
        0.0,
    ))
    .with_text_options(TextContentsOptions::default().with_max_characters(12));
let hits = search.search("rust");
assert_eq!(hits[0].title, "Rust async guide");

let similar = ExaFindSimilarResults::new().with_hit(SearchHit::new(
    "Rust async guide",
    "https://example.com/rust",
    "Rust async runtime and futures",
    0.0,
));
assert_eq!(similar.find_similar("async runtime").len(), 1);

let retriever = ExaSearchRetriever::new()
    .with_max_results(1)
    .with_hit(SearchHit::new(
        "LangChain",
        "https://example.com/langchain",
        "Build retrieval pipelines in Rust",
        0.0,
    ));
```

## Public Surface

- `ExaSearchRetriever`
- `ExaSearchResults`, `ExaFindSimilarResults`, `SearchHit`
- `TextContentsOptions`, `HighlightsContentsOptions`
- `retrievers`, `tools`, `types` namespaces

## Tests

- `tests/retriever_and_tools.rs`
- `tests/namespace.rs`
