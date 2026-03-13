# langchain-ollama

Ollama chat, LLM, and embeddings integration for local model runtimes.

## Upstream Mapping

This crate maps to `libs/partners/ollama` in the reference monorepo.

## Installation

```bash
cargo add langchain-ollama
```

## Quick Start

```rust
use langchain_ollama::{ChatOllama, OllamaEmbeddings, OllamaLLM};

let chat = ChatOllama::new("llama3.1", None::<&str>);
let llm = OllamaLLM::new("llama3.1", None::<&str>);
let embeddings = OllamaEmbeddings::new("nomic-embed-text", None::<&str>);

assert_eq!(chat.base_url(), "http://localhost:11434/v1");
assert_eq!(llm.base_url(), "http://localhost:11434/v1");
assert_eq!(embeddings.base_url(), "http://localhost:11434/v1");
```

## Public Surface

- `ChatOllama`
- `OllamaLLM`
- `OllamaEmbeddings`
- `chat_models`, `llms`, `embeddings` namespaces

## Tests

- `tests/provider.rs`
- `tests/namespace.rs`

