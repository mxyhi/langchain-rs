# langchain-core

Core abstractions and protocol surfaces for the Rust LangChain workspace.

## Upstream Mapping

This crate maps to `libs/core` in the reference monorepo.

## Installation

```bash
cargo add langchain-core
```

## Quick Start

```rust
use langchain_core::messages::HumanMessage;
use langchain_core::prompt_values::StringPromptValue;

let prompt = StringPromptValue::new("hello");
let message = HumanMessage::new(prompt.to_string());
assert_eq!(message.content(), "hello");
```

## Public Surface

- `messages`, `prompt_values`, `prompts`, `language_models`, `embeddings`
- `document_loaders`, `retrievers`, `vectorstores`, `stores`, `structured_query`
- `callbacks`, `tracers`, `tools`, `load`, `chat_history`, `chat_sessions`
- `utils::{env, strings, iter, aiter, usage, uuid, utils, _merge}` for shared helper parity with the reference package

## Tests

- `tests/namespace_surface.rs`
- `tests/messages_parity.rs`
- `tests/tools_and_parsers.rs`
- `tests/utils_extensions.rs`
