# langchain-huggingface

Hugging Face chat, endpoint, pipeline, and embeddings integrations.

## Upstream Mapping

This crate maps to `libs/partners/huggingface` in the reference monorepo.

## Installation

```bash
cargo add langchain-huggingface
```

## Quick Start

```rust
use langchain_huggingface::{
    ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint, HuggingFacePipeline,
};

let chat = ChatHuggingFace::new_with_base_url(
    "meta-llama/Llama-3.1-8B-Instruct:hf-inference",
    "https://example.com/v1",
    Some("hf-test"),
);
let endpoint = HuggingFaceEndpoint::new("mistralai/Mistral-7B-Instruct-v0.3");
let pipeline = HuggingFacePipeline::new("meta-llama/Llama-3.1-8B-Instruct");
let embeddings = HuggingFaceEmbeddings::new("sentence-transformers/all-mpnet-base-v2");

assert_eq!(chat.model_name(), "meta-llama/Llama-3.1-8B-Instruct:hf-inference");
assert_eq!(endpoint.model_name(), "mistralai/Mistral-7B-Instruct-v0.3");
assert_eq!(pipeline.model_name(), "meta-llama/Llama-3.1-8B-Instruct");
assert_eq!(embeddings.model_name(), "sentence-transformers/all-mpnet-base-v2");
```

## Public Surface

- `ChatHuggingFace`
- `HuggingFaceEmbeddings`, `HuggingFaceEndpointEmbeddings`
- `HuggingFaceEndpoint`, `HuggingFacePipeline`
- `chat_models`, `embeddings`, `llms` namespaces

## Tests

- `tests/boundaries.rs`
- `tests/namespace.rs`

