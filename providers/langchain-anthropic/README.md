# langchain-anthropic

Anthropic chat model and LLM integration with Anthropic-native tool schemas.

## Upstream Mapping

This crate maps to `libs/partners/anthropic` in the reference monorepo.

## Installation

```bash
cargo add langchain-anthropic
```

## Quick Start

```rust
use langchain_anthropic::{AnthropicLLM, ChatAnthropic};

let chat = ChatAnthropic::new("claude-3-7-sonnet-latest", "https://api.anthropic.com", Some("test-key"));
let llm = AnthropicLLM::new("claude-3-7-sonnet-latest", "https://api.anthropic.com", None::<&str>);

assert_eq!(chat.model_name(), "claude-3-7-sonnet-latest");
assert_eq!(llm.model_name(), "claude-3-7-sonnet-latest");
```

## Public Surface

- `ChatAnthropic`
- `AnthropicLLM`
- `AnthropicToolDefinition`, `convert_to_anthropic_tool`
- `data::anthropic_profile()` for provider profile metadata
- `chat_models`, `llms`, `middleware`, `output_parsers`, `experimental` namespaces

## Tests

- `tests/chat_model.rs`
- `tests/llms.rs`
- `tests/middleware_and_experimental.rs`
- `tests/namespace.rs`
