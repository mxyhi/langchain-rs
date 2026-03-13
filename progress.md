# Progress

## 2026-03-13

### Session start

- 目标：一次性完成对 `.ref/langchain` 可映射公开接口、文档和验证面的收口。
- 约束：先调研后落地；复杂任务根目录落盘；多 agent 并行；默认自主决策，不中途停等用户确认。
- 已完成：
  - 读取 `planning-with-files`、`brainstorming`、`test-driven-development` 技能。
  - 读取 workspace `Cargo.toml`、根 `README.md`、关键 crate `lib.rs`。
  - 读取 `.ref/langchain` 顶层与关键 libs README。
  - 用 DeepWiki 校验参考 monorepo 各 package 职责边界。
- 下一步：
  - 生成设计文档并提交基线 commit。
  - 派发并行 agent 做 surface/test/provider 差异盘点。
  - 建立 failing tests / 文档检查并开始实现。

### Design review

- 设计文档经独立审阅后已修正三点：
  - provider 不再一律按聊天模型描述，而是按能力类型区分 `chat/llm/embeddings/vectorstore/retriever/tooling`。
  - 设计中单列 `langchain-text-splitters`、`langchain-model-profiles`、`langchain-tests` 的顶级职责。
  - 文档验证机制明确为 “README 断言 + doc comment / integration tests 可编译示例”。

### Implementation summary

- 已完成根 README 和全部公开 crate/provider README，新增 `crates/langchain/tests/readme_parity.rs` 并先跑红后转绿。
- 已收口 provider namespace parity：
  - `langchain-anthropic` 公开 `chat_models` / `llms`
  - `langchain-openai` 公开 `azure` / `chat_models` / `compatible` / `embeddings` / `llms` / `tools`
  - `langchain-exa` 公开 `retrievers` / `tools` / `types`
  - `langchain-perplexity` 补齐 `tests/namespace.rs`
- 已收口 `langchain-tests` harness：新增 `unit_tests.rs`、`integration_tests.rs`、`tests/standard_harness.rs`
- 已收口 `langchain-model-profiles refresh`：支持 `--provider` / `--data-dir` / `--catalog`，输出 `_profiles.json`
- 过程中遇到两次 `langchain-model-profiles` 局部编译问题：
  - `BTreeMap` 类型推断瞬态报错，后续确认代码已显式标注并通过单包测试
  - `reqwest::blocking` feature 不可用，最终改回 `tokio` runtime + `reqwest::get` 方案

### Verification

- `cargo test -p langchain --test readme_parity --quiet`
- `cargo test -p langchain-tests --quiet`
- `cargo test -p langchain-model-profiles --quiet`
- `cargo test -p langchain-anthropic --quiet`
- `cargo test -p langchain-openai --quiet`
- `cargo test -p langchain-exa --quiet`
- `cargo test -p langchain-perplexity --quiet`
- `cargo fmt --all`
- `cargo test --workspace --quiet`
