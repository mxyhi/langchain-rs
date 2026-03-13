# langchain-rs parity plan

## Goal

一次性完成当前 Rust workspace 对 `.ref/langchain` 参考面的可映射 1:1 复刻，包括公开接口、文档叙述、示例/测试支撑和可验证的 workspace 完整性。

## Scope Defaults

- 以 `.ref/langchain/libs/{core,langchain,langchain_v1,model-profiles,standard-tests,text-splitters,partners/*}` 为参考面。
- 只对齐当前 workspace 已声明的 crate/provider；不凭空扩张到 workspace 之外的上游包。
- `langchain-core` 对应基础抽象层，`langchain` 对应 v1 facade，`langchain-classic` 对应 classic/legacy，provider crate 对应 partners。
- 文档至少覆盖根 README、每个公开 crate/provider README、必要的 doctest/usage 示例。
- 以当前 Rust 语言约束为准，不为兼容旧格式保留不必要负担。

## Phases

| Phase | Status | Notes |
| --- | --- | --- |
| 1. 调研参考面与现状 | in_progress | 已确认包映射与 monorepo 边界 |
| 2. 根目录设计/计划落盘 | in_progress | 本文件、findings、progress、design doc |
| 3. 差异盘点与并行拆分 | pending | surface、tests、docs、providers |
| 4. 先测后改 | pending | 缺口先转成 failing test 或缺文档检查 |
| 5. 实现与文档补齐 | pending | 代码、README、示例 |
| 6. 全量验证与提交 | pending | cargo test / fmt / git commit |

## Success Criteria

- workspace 公开 crate/provider 的 README 与参考仓语义对齐。
- 核心 facade/classic/core/text-splitters/model-profiles/standard-tests 暴露面与当前参考映射一致。
- provider crate 拥有最小可用公开入口与文档。
- 全量测试通过，差异点记录在 `findings.md` 和 `progress.md`。
- 形成至少一个可追溯 commit。

## Risks

- “1:1” 无法逐字符等价，只能以 crate/package 边界、公开接口语义、测试与文档覆盖为准。
- provider 覆盖面大，若某些 crate 只缺文档则优先补文档和导出面，不做无证据的功能扩张。
