# langchain-rs

Rust workspace that mirrors the `.ref/langchain` monorepo with Rust-native crate boundaries.

## Workspace Layout

| Rust crate | Reference package | Role |
| --- | --- | --- |
| `crates/langchain-core` | `libs/core` | Core abstractions, protocols, loaders, stores, tools, tracers, vectorstores |
| `crates/langchain` = `libs/langchain_v1` | `libs/langchain_v1` | v1-style facade, factories, agents, middleware, provider routing |
| `crates/langchain-classic` = `libs/langchain` | `libs/langchain` | Classic compatibility surface for legacy chains, memory, utilities |
| `crates/langchain-text-splitters` | `libs/text-splitters` | Standalone text splitter package |
| `crates/langchain-model-profiles` | `libs/model-profiles` | Provider profile registry and refresh tooling |
| `crates/langchain-tests` | `libs/standard-tests` | Reusable standard test helpers and harnesses |
| `providers/*` | `libs/partners/*` | Provider-specific model, embedding, vectorstore, retriever, and tooling crates |

## Quick Start

```bash
cargo test --workspace
```

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

## Provider Matrix

Current provider crates in this workspace:

- `langchain-openai`, `langchain-anthropic`, `langchain-deepseek`, `langchain-fireworks`, `langchain-groq`, `langchain-huggingface`, `langchain-mistralai`, `langchain-nomic`, `langchain-ollama`, `langchain-openrouter`, `langchain-perplexity`, `langchain-xai`
- `langchain-chroma`, `langchain-qdrant`
- `langchain-exa`

The mix is intentionally heterogeneous: some crates expose chat models and LLMs, some focus on embeddings, and others are vectorstore or retriever/tool integrations.

## Development

- Run the full test matrix with `cargo test --workspace`.
- Use crate-level READMEs under `crates/*` and `providers/*` for package-specific quick starts and public-surface summaries.
- Use `.ref/langchain` to compare crate boundaries and naming with the reference monorepo.
