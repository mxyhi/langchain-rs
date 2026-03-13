# langchain-groq

Groq chat model integration with provider-default routing.

## Upstream Mapping

This crate maps to `libs/partners/groq` in the reference monorepo.

## Installation

```bash
cargo add langchain-groq
```

## Quick Start

```rust
use langchain_groq::ChatGroq;

let model = ChatGroq::new("llama-3.1-8b-instant", None::<&str>);
assert_eq!(model.base_url(), "https://api.groq.com/openai/v1");
```

## Public Surface

- `ChatGroq`
- `data::groq_profile()` for provider profile metadata
- `chat_models::ChatGroq`
- Groq default base URL handling

## Tests

- `tests/chat_model.rs`
- `tests/namespace.rs`
