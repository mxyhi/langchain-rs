# Findings

## 2026-03-13

### Workspace and reference mapping

- 当前 workspace 成员与 `.ref/langchain/libs` 高度同构：`langchain-core`、`langchain`、`langchain-classic`、`langchain-model-profiles`、`langchain-tests`、`langchain-text-splitters` 以及 13 个 partner/provider crate。
- `.ref/langchain` 是 Python monorepo，本地 `README` 与各 `libs/*/README.md` 明确了包职责；DeepWiki 结果进一步确认 `core` 是基础抽象层，`langchain_v1` 是主入口，`langchain-classic` 是 legacy 边界，`model-profiles` 和 `standard-tests` 更偏维护/测试工具层。
- 当前根 `README.md` 只有一句 `1:1 made langchain in rust`，明显不足以支撑“接口、文档等全部功能”的交付目标。

### Current codebase state

- `crates/langchain/src/lib.rs` 已提供 facade 风格入口：`init_chat_model`、`init_configurable_chat_model`、`init_embeddings`，并重导出 core/text-splitters/tools/vectorstores 等命名空间。
- `crates/langchain-classic/src/lib.rs` 已承载 classic 兼容面，包含 `chains`、`memory`、`utilities`、`text_splitter` 等 legacy 入口。
- 最近提交已连续推进 parity：`feat(parity): expand core classic and text splitter surfaces`、`feat(providers): add remote transports and provider namespaces`、`feat(langchain): add middleware parity surface` 等。

### Documentation gaps

- 仓库根目录暂无 `docs/`，公开 crate/provider 目录下也未发现 README。
- 需要补齐根 README 与 crate README，至少说明职责、安装方式、快速示例、与参考仓映射关系。

### Implemented closure

- README parity 已补齐：根 README + 6 个 crate README + 15 个 provider README 全部落地，并通过新增的 `crates/langchain/tests/readme_parity.rs` 自动校验。
- `langchain-model-profiles` 不再只是静态展示 CLI，现已支持 `refresh --provider --data-dir [--catalog]`，可从 `models.dev` catalog 或本地 fixture 生成 `_profiles.json`。
- `langchain-tests` 从单个 assert helper 提升为可复用 harness：新增 unit/integration test runners，保留原 helper 作为底层断言。
- provider namespace parity 已补齐最小闭包：`langchain-anthropic`、`langchain-openai`、`langchain-exa` 新增公开 namespace；`langchain-perplexity` 补齐 namespace parity tests。
- `langchain-classic` 顶层注释已去掉“intentionally minimal for now”这一与当前目标不一致的表述。
