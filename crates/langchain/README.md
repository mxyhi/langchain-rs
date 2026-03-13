# langchain

High-level facade crate for the Rust workspace, corresponding to the v1 LangChain package.

## Upstream Mapping

This crate maps to `libs/langchain_v1` in the reference monorepo.

## Installation

```bash
cargo add langchain
```

## Quick Start

```rust
use langchain::messages::HumanMessage;
use langchain::runnables::Runnable;

# async fn demo() -> Result<(), langchain::LangChainError> {
let model = langchain::init_chat_model(
    "openai:gpt-4o-mini",
    langchain::ModelInitOptions::default()
        .with_base_url("https://api.openai.com/v1")
        .with_api_key("test-key"),
)?;
let _message = model
    .invoke(vec![HumanMessage::new("ping").into()], Default::default())
    .await?;
# Ok(())
# }
```

## Public Surface

- Factory entry points: `ModelInitOptions`, `init_chat_model`, `init_configurable_chat_model`, `init_embeddings`
- Facade namespaces: `agents`, `chat_models`, `embeddings`, `messages`, `prompts`, `tools`, `vectorstores`
- Re-exported core and splitter abstractions for application-facing usage

## Tests

- `tests/facade.rs`
- `tests/factories.rs`
- `tests/agents.rs`
- `tests/middleware.rs`

