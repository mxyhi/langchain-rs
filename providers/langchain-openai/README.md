# langchain-openai

OpenAI chat, LLM, embeddings, Azure, and OpenAI-compatible transport integration.

## Upstream Mapping

This crate maps to `libs/partners/openai` in the reference monorepo.

## Installation

```bash
cargo add langchain-openai
```

## Quick Start

```rust
use langchain_openai::{ChatOpenAI, OpenAI, OpenAIEmbeddings};

let chat = ChatOpenAI::new("gpt-4o-mini", "https://api.openai.com/v1", Some("test-key"));
let llm = OpenAI::new("gpt-4o-mini", "https://api.openai.com/v1", None::<&str>);
let embeddings = OpenAIEmbeddings::new("text-embedding-3-small", "https://api.openai.com/v1", None::<&str>);

assert_eq!(chat.model_name(), "gpt-4o-mini");
assert_eq!(llm.model_name(), "gpt-4o-mini");
assert_eq!(embeddings.model_name(), "text-embedding-3-small");
```

## Public Surface

- `ChatOpenAI`, `OpenAI`, `OpenAIEmbeddings`
- `AzureChatOpenAI`, `AzureOpenAI`, `AzureOpenAIEmbeddings`
- `OpenAICompatibleChatModel`, `OpenAICompatibleEmbeddings`, `OpenAICompatibleLlm`
- `custom_tool` and namespace modules `chat_models`, `embeddings`, `llms`, `compatible`, `azure`, `tools`

## Tests

- `tests/chat_model.rs`
- `tests/embeddings.rs`
- `tests/llms.rs`
- `tests/structured_output.rs`
- `tests/namespace.rs`

