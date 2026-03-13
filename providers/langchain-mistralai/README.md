# langchain-mistralai

MistralAI chat and embeddings integration with default provider routing.

## Upstream Mapping

This crate maps to `libs/partners/mistralai` in the reference monorepo.

## Installation

```bash
cargo add langchain-mistralai
```

## Quick Start

```rust
use langchain_mistralai::{ChatMistralAI, MistralAIEmbeddings};

let chat = ChatMistralAI::new("mistral-small-latest", None::<&str>);
let embeddings = MistralAIEmbeddings::new("mistral-embed", None::<&str>);

assert_eq!(chat.base_url(), "https://api.mistral.ai/v1");
assert_eq!(embeddings.base_url(), "https://api.mistral.ai/v1");
```

## Public Surface

- `ChatMistralAI`
- `MistralAIEmbeddings`
- `data::mistralai_profile()` for provider profile metadata
- `chat_models`, `embeddings` namespaces

## Tests

- `tests/provider.rs`
- `tests/namespace.rs`
