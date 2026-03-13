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
