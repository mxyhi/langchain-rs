# langchain-fireworks

Fireworks chat, LLM, and embeddings integration with OpenAI-compatible transports.

## Upstream Mapping

This crate maps to `libs/partners/fireworks` in the reference monorepo.

## Installation

```bash
cargo add langchain-fireworks
```

## Quick Start

```rust
use langchain_fireworks::{ChatFireworks, Fireworks, FireworksEmbeddings};

let chat = ChatFireworks::new(
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
    None::<&str>,
);
let llm = Fireworks::new(
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
    None::<&str>,
);
let embeddings = FireworksEmbeddings::new("nomic-ai/nomic-embed-text-v1.5", None::<&str>);

assert_eq!(chat.base_url(), "https://api.fireworks.ai/inference/v1");
assert_eq!(llm.base_url(), "https://api.fireworks.ai/inference/v1");
assert_eq!(embeddings.base_url(), "https://api.fireworks.ai/inference/v1");
```

## Public Surface

- `ChatFireworks`
- `Fireworks`
- `FireworksEmbeddings`
- `chat_models`, `llms`, `embeddings` namespaces

## Tests

- `tests/provider.rs`
- `tests/namespace.rs`
- `tests/transport.rs`
