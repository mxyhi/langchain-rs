# langchain-xai

xAI chat model integration with provider-default routing.

## Upstream Mapping

This crate maps to `libs/partners/xai` in the reference monorepo.

## Installation

```bash
cargo add langchain-xai
```

## Quick Start

```rust
use langchain_xai::ChatXAI;

let model = ChatXAI::new("grok-3-mini", None::<&str>);
assert_eq!(model.base_url(), "https://api.x.ai/v1");
```

## Public Surface

- `ChatXAI`
- `chat_models::ChatXAI`
- xAI default base URL handling

## Tests

- `tests/chat_model.rs`
- `tests/namespace.rs`
