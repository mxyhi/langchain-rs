# langchain-tests

Reusable standard-test helpers and harnesses for integration crates in the workspace.

## Upstream Mapping

This crate maps to `libs/standard-tests` in the reference monorepo.

## Installation

```bash
cargo add langchain-tests --dev
```

## Quick Start

```rust
use langchain_core::messages::{BaseMessage, HumanMessage};
use langchain_core::language_models::ParrotChatModel;
use langchain_tests::assert_chat_model_response;

# async fn demo() {
let model = ParrotChatModel::new("parrot-1", 5);
assert_chat_model_response(
    &model,
    vec![BaseMessage::from(HumanMessage::new("hello"))],
    "hello",
)
.await;
# }
```

## Public Surface

- Reusable assertions for chat models, LLMs, embeddings, retrievers, and vectorstores
- Reusable assertions for tools, caches, and stores
- Standardized test harness modules: `unit_tests` and `integration_tests`
- `BaseStandardTests` and `StandardTestSuite` helpers for guarding standard-suite drift
- Standard suites for caches, base stores, tools, document indexes, and sandboxes
- Utility compatibility surface such as `utils::pydantic`
- Shared test-only traits such as `EmbeddingsUnderTest` and `VectorStoreUnderTest`

## Tests

- `tests/standard_helpers.rs`
- `tests/tools_cache_store.rs`
- `tests/extended_standard_helpers.rs`
- `tests/standard_surface.rs`
