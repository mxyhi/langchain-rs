# LangChain Rust V1 Boundary Parity Design

## Context

`langchain-rs` 已经完成 provider parity milestone，但这只覆盖了部分 provider crate 和 `langchain` factory。

参考仓 `.ref/langchain` 显示，真正的 `langchain` 1.x 对外入口还包括：

- `langchain.messages`
- `langchain.tools`
- `langchain.chat_models`
- `langchain.embeddings`
- `langchain.agents`
- `langchain.rate_limiters`
- `langchain_classic` 作为独立 legacy/classic package boundary

当前 Rust workspace 的主要缺口不是 provider 数量，而是 v1 facade/core/classic 的公共边界仍然偏薄。

## Goal

在本轮一次性交付中，把 Rust workspace 推进到“v1 public boundary parity”：

- `langchain-core` 补齐 v1 facade 依赖的核心消息/工具/限流抽象；
- `langchain` 补齐 `agents`、`rate_limiters`、更完整的 `messages/tools/chat_models/embeddings` 对外入口；
- `langchain-classic` 从 marker crate 升级为真实 package boundary；
- provider crate 继续对齐参考仓顶层导出名，避免 façade 指向不存在的 public API；
- 所有新增边界都通过测试锁定，不做空喊口号式的“1:1”。

## Non-Goals

- 不在本轮伪造 Python monorepo 的全部高级行为，如完整 middleware graph、LangGraph 深度集成、社区模块全集。
- 不为当前 workspace 不存在的 provider crate 发明虚假实现。
- 不牺牲可读性引入大而全的兼容层。

## Design

### 1. `langchain-core` 先补最小公共积木

新增或补强以下边界：

- richer message exports: 内容块、chunk/server-tool 相关类型、`AnyMessage`、`trim_messages`
- tool utility surface: `ToolException` 与必要的注入型标记边界
- rate limiters: `BaseRateLimiter`、`InMemoryRateLimiter`

这些类型优先保持 Rust 风格，但命名与用途对齐参考仓。

### 2. `langchain` facade 补齐 v1 入口

新增：

- `langchain::agents`
- `langchain::rate_limiters`

其中 `agents` 采用“诚实的最小可用实现”：

- 暴露 `AgentState`
- 暴露 `create_agent`
- 暴露 structured-output 相关错误类型

行为上只覆盖当前 workspace 真正能支撑的单模型/单轮 agent 调用，不虚构 LangGraph 工作流。

### 3. `langchain-classic` 改成真实 boundary crate

`langchain-classic` 至少暴露一层稳定模块边界，例如：

- `chat_models`
- `embeddings`
- `llms`
- `messages`
- `prompts`
- `runnables`
- `tools`
- `vectorstores`

能直接复用现有 core/facade 的地方就 re-export；需要 legacy 语义但当前未实现的地方，给出明确的 unsupported/placeholder 边界，而不是完全缺失模块。

### 4. provider crates 继续收口顶层导出

对照 `.ref/langchain/libs/partners/*/langchain_*`：

- 补齐根模块 `pub use` 缺失
- 对“只能 honest unsupported”的类型给出真实 public type 和测试
- 避免 facade 或 registry 声明的导出名在 crate root 不可见

## Testing

- 先补失败测试，再写实现
- 核心覆盖：
  - facade/classic/re-export 边界
  - message/rate limiter/agent 最小行为
  - provider 根导出存在性
  - 全量 `cargo test`

## Acceptance

- `langchain` 与 `langchain-classic` 拥有可导入、可测试的 v1 boundary
- `langchain-core` 拥有 v1 facade 依赖的最小公共抽象
- provider crate 的根导出与参考矩阵不再明显失配
- `cargo test` 全绿
