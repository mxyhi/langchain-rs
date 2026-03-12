# LangChain Rust Provider Parity Design

## Context

`langchain-rs` currently mirrors the Python monorepo package layout, but only `langchain-openai` contains working behavior. The remaining provider crates are placeholders, and the `langchain` factory only supports `openai`.

The reference monorepo under `.ref/langchain` shows that the current workspace boundary should cover:

- `langchain-core`
- `langchain`
- `langchain-classic`
- `langchain-model-profiles`
- `langchain-tests`
- `langchain-text-splitters`
- partner crates under `providers/`

The most credible one-shot delivery for this phase is not a fake claim of full Python parity. It is to turn the already-created workspace crates into real, testable Rust APIs whose public surfaces line up with the reference package exports and factory rules.

## Goal

Deliver a provider-parity milestone that:

- removes placeholder-only provider crates for the currently declared workspace members,
- adds real public types matching the reference partner package export matrix,
- wires the `langchain` factory to supported providers with inference and default base URLs,
- centralizes provider metadata in `langchain-model-profiles`,
- keeps the implementation simple and heavily shared where providers are protocol-compatible,
- validates the result through targeted tests plus full `cargo test`.

## Non-Goals

- Full port of every Python monorepo module in one turn.
- Full tracing/callback/history/indexing parity with Python `langchain_core`.
- Provider-specific advanced features whose transport or runtime model differs substantially from the existing Rust baseline.

## Design

### 1. Shared provider metadata

`langchain-model-profiles` becomes the single source of truth for:

- provider keys,
- default base URLs,
- model prefix inference for chat models,
- provider capability exposure across chat/llm/embeddings/vector stores.

This prevents factory logic and provider crates from hard-coding divergent defaults.

### 2. OpenAI-compatible provider family

Several providers in the reference boundary expose APIs that can be represented by the existing `langchain-openai` transport model for this milestone. For these crates, we will create thin typed wrappers over the existing OpenAI-compatible implementations and only vary:

- default base URL,
- exported type names,
- supported capabilities.

This applies only to the provider family where the current milestone can credibly share request/response handling without introducing fake bespoke behavior:

- `deepseek`
- `fireworks`
- `groq`
- `mistralai`
- `openrouter`
- `xai`

`ollama` may share parts of the interface shape, but its initialization and wire behavior are provider-specific enough that it must keep an explicit wrapper boundary rather than a blind alias.

### 3. Non-compatible providers

`langchain-anthropic` gets its own minimal chat/llm client because its wire contract is materially different from the OpenAI-compatible family.

`langchain-qdrant` and `langchain-chroma` stop being pure descriptors and instead expose real vector store boundaries aligned with their reference package roles. The initial milestone focuses on public API and in-memory/testable behavior rather than network-backed completeness.

`langchain-exa` is also in scope. It must stop being a pure descriptor and expose retriever/tool-oriented public types aligned with the reference package role.

`langchain-perplexity` must expose its chat model plus its retriever/tool/parser public surface. The milestone may keep advanced behavior minimal, but the crate cannot remain a pure descriptor.

`langchain-huggingface` and `langchain-nomic` are also in scope. Their implementations may be minimal boundary-first adapters in this milestone, but they must expose the exact top-level names from the reference package and cannot remain marker-only placeholders.

### 4. Factory parity

`crates/langchain` will use the shared provider registry for:

- `init_chat_model`,
- `init_configurable_chat_model`,
- `init_embeddings`,
- model prefix inference,
- consistent supported-provider errors.

The base URL resolution order is explicit and must be tested:

1. explicit caller-supplied `base_url`,
2. provider-specific runtime override from `RunnableConfig` where applicable,
3. provider default from `langchain-model-profiles`.

## Export Matrix

This milestone treats the following top-level exports as the minimum public surface to align:

- `langchain-openai`: `ChatOpenAI`, `OpenAIEmbeddings`, `OpenAI`
- `langchain-anthropic`: `ChatAnthropic`, `AnthropicLLM`, `convert_to_anthropic_tool`
- `langchain-ollama`: `ChatOllama`, `OllamaEmbeddings`, `OllamaLLM`
- `langchain-deepseek`: `ChatDeepSeek`
- `langchain-fireworks`: `ChatFireworks`, `FireworksEmbeddings`, `Fireworks`
- `langchain-groq`: `ChatGroq`
- `langchain-huggingface`: `ChatHuggingFace`, `HuggingFaceEmbeddings`, `HuggingFaceEndpointEmbeddings`, `HuggingFaceEndpoint`, `HuggingFacePipeline`
- `langchain-mistralai`: `ChatMistralAI`, `MistralAIEmbeddings`
- `langchain-nomic`: `NomicEmbeddings`
- `langchain-openrouter`: `ChatOpenRouter`
- `langchain-perplexity`: `ChatPerplexity`, `PerplexitySearchRetriever`, `PerplexitySearchResults`, `UserLocation`, `WebSearchOptions`, `MediaResponse`, `MediaResponseOverrides`, `ReasoningJsonOutputParser`, `ReasoningStructuredOutputParser`, `strip_think_tags`
- `langchain-qdrant`: `FastEmbedSparse`, `Qdrant`, `QdrantVectorStore`, `RetrievalMode`, `SparseEmbeddings`, `SparseVector`
- `langchain-chroma`: `Chroma`
- `langchain-xai`: `ChatXAI`
- `langchain-exa`: `ExaSearchRetriever`, `ExaSearchResults`, `ExaFindSimilarResults`

## Testing

- Add failing tests first for each new factory/provider behavior.
- Cover provider inference, explicit provider selection, default base URLs, and public re-exports.
- Keep full `cargo test` green.

## Acceptance

- No placeholder-only provider crates remain among the declared workspace members.
- `langchain` factory supports the provider set implemented in this milestone.
- Shared metadata lives in `langchain-model-profiles`.
- Tests demonstrate provider export coverage, base URL resolution priority, and pass in the full workspace run.
