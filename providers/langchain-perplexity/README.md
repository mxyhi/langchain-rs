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
use langchain_core::language_models::BaseChatModel;
use langchain_perplexity::{ChatPerplexity, WebSearchOptions};

let model = ChatPerplexity::new("sonar")
    .with_web_search_options(WebSearchOptions::default().with_search_context_size(3));
assert_eq!(model.model_name(), "sonar");
```

## Public Surface

- `ChatPerplexity`
- `ReasoningJsonOutputParser`, `ReasoningStructuredOutputParser`, `strip_think_tags`
- `PerplexitySearchRetriever`, `PerplexitySearchResults`, `PerplexitySearchHit`
- `UserLocation`, `MediaResponse`, `MediaResponseOverrides`, `WebSearchOptions`
- `data::perplexity_profile()`, `data::perplexity_exports()`, `data::default_base_url()`
- `chat_models`, `retrievers`, `tools`, `output_parsers`, `types`, `data` namespaces

## Tests

- `tests/boundaries.rs`
- `tests/namespace.rs`
