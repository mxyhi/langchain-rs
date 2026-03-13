# langchain-perplexity

Perplexity chat, reasoning parser, retriever, and search-tool integration.

## Upstream Mapping

This crate maps to `libs/partners/perplexity` in the reference monorepo.

## Installation

```bash
cargo add langchain-perplexity
```

## Quick Start

```rust
use langchain_perplexity::{ChatPerplexity, WebSearchOptions};

let model = ChatPerplexity::new("sonar", None::<&str>)
    .with_web_search_options(WebSearchOptions::default().with_search_context_size(3));
assert_eq!(model.model_name(), "sonar");
```

## Public Surface

- `ChatPerplexity`
- `ReasoningJsonOutputParser`, `ReasoningStructuredOutputParser`, `strip_think_tags`
- `PerplexitySearchRetriever`, `PerplexitySearchResults`, `PerplexitySearchHit`
- `chat_models`, `retrievers`, `tools`, `output_parsers`, `types` namespaces

## Tests

- `tests/boundaries.rs`
- `tests/namespace.rs`
