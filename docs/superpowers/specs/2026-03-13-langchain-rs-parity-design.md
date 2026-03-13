# langchain-rs parity design

## Context

当前仓库已搭出与 `.ref/langchain` 对应的 Rust workspace，但交付状态仍偏“代码骨架 + 局部测试”，与“1:1 复刻接口、文档等全部功能”之间的主要差距不在 monorepo 结构，而在三个方面：

1. 根 README 和 crate/provider README 几乎为空，无法表达 package 职责、快速用法与参考映射。
2. 公开 surface 需要再做一次基于参考仓职责边界的核对，确认 facade/classic/core/provider 各自承接的命名空间没有漏口。
3. 需要把这些边界写进测试与验证链路，避免只靠人工判断“像不像”。

## Default Decisions

- 采用“按当前 workspace 成员做强对齐”的方案，不跨出已声明 crates/providers 去扩张其它参考包。
- 保持分层结构：
  - `langchain-core`：基础抽象与协议。
  - `langchain`：v1 风格 facade、agents、provider 初始化入口。
  - `langchain-classic`：legacy/classic compatibility surface。
- `langchain-text-splitters`：独立文本切分工具包。
- `langchain-model-profiles`：provider profile 元数据与维护 CLI。
- `langchain-tests`：标准化集成测试辅助层。
- provider crate 按 provider 类型暴露对应的 `chat model` / `llm` / `embeddings` / `vectorstore` / `retriever` / `tooling` API，并提供一致的初始化入口与 README usage。
- 文档采用“根 README 总览 + 每 crate README 说明职责/安装/示例/映射”的最小完整集。
- 验证优先顺序为：现有测试基线 -> 新增 parity/doc tests -> 全 workspace `cargo test`。

## Implementation Plan

### 1. Surface audit

基于 `.ref/langchain/libs/*/README.md` 和当前各 crate `lib.rs`，生成缺口清单。重点看：

- `langchain-core` 是否完整导出参考中的基础命名空间。
- `langchain` 是否承接 v1 facade、模型初始化、middleware/tooling 入口。
- `langchain-classic` 是否承接 classic 专属入口。
- provider crate 是否具备一致的公开入口与最小文档。

### 2. Test-first closure

对接口缺口先补 failing tests。对文档缺口，采用两层约束确保能回归验证：

- README 存在性和关键片段测试，覆盖根 README 与公开 crate/provider README。
- 将 README 中的最小使用片段同步为 crate 级 doc comment 示例或现有 integration tests 中的可编译片段，避免 README 只“看起来存在”。

### 3. Documentation parity

补齐根 README 和 crate/provider README，统一说明：

- 对应上游包。
- Rust crate 的职责。
- 快速使用示例。
- 与其它 crate 的关系。

### 4. Final verification

执行格式化与全量测试，确认 workspace 在 clean 状态下可通过，然后提交一个收口 commit。

## Acceptance

- 根 README 可单独解释整个 workspace 的职责分层与 provider 版图。
- 公开 crate/provider 目录具备 README。
- surface tests 和现有行为测试一起通过。
- 最终差异与边界写入 `findings.md` / `progress.md`。
