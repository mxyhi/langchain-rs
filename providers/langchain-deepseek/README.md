# langchain-deepseek

DeepSeek chat model integration built on the workspace provider facade.

## Upstream Mapping

This crate maps to `libs/partners/deepseek` in the reference monorepo.

## Installation

```bash
cargo add langchain-deepseek
```

## Quick Start

```rust
use langchain_deepseek::ChatDeepSeek;

let model = ChatDeepSeek::new("deepseek-chat", None::<&str>);
assert_eq!(model.base_url(), "https://api.deepseek.com/v1");
```

## Public Surface

- `ChatDeepSeek`
- `data::deepseek_profile()` for provider profile metadata
- `chat_models::ChatDeepSeek`
- DeepSeek-compatible default base URL routing

## Tests

- `tests/chat_model.rs`
- `tests/namespace.rs`
