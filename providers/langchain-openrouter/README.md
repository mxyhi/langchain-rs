# langchain-openrouter

OpenRouter chat model integration with reference default routing.

## Upstream Mapping

This crate maps to `libs/partners/openrouter` in the reference monorepo.

## Installation

```bash
cargo add langchain-openrouter
```

## Quick Start

```rust
use langchain_openrouter::ChatOpenRouter;

let model = ChatOpenRouter::new("openai/gpt-4o-mini", None::<&str>);
assert_eq!(model.base_url(), "https://openrouter.ai/api/v1");
```

## Public Surface

- `ChatOpenRouter`
- `data::openrouter_profile()` for provider profile metadata
- `chat_models::ChatOpenRouter`
- OpenRouter default base URL handling

## Tests

- `tests/chat_model.rs`
- `tests/namespace.rs`
