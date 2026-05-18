# AgentLab v0.4.2 Release Notes

发布日期：2026-05-18

## 版本概览

v0.4.2 聚焦 CLI 与配置可观测性增强，不改变核心研究流程输出语义。

## 新增能力

1. `--config`

- 支持通过 YAML 文件加载运行配置。
- 推荐以 `docs/agentlab.sample.yaml` 为模板。

2. `--list-strategy-presets`

- 可列出内置策略预设，便于快速选择策略风格。
- 当前内置：`default`、`concise`。

3. `--print-effective-config`

- 在运行前输出“最终生效配置”，用于排查配置来源与覆盖关系。

## 配置优先级（重要）

生效顺序固定为：

`CLI 参数 > 配置文件(--config) > 默认值`

这意味着同一字段若同时在 CLI 与配置文件中出现，CLI 值优先。

## 使用示例

```bash
uv run python examples/04_deep_research.py \
  "研究 LangGraph、AutoGen、CrewAI 的区别" \
  --config docs/agentlab.sample.yaml \
  --print-effective-config
```

```bash
uv run python examples/04_deep_research.py --list-strategy-presets
```

## 兼容性说明

- 无破坏性变更。
- 现有命令可继续使用。
- 输出文件路径与命名保持不变（`outputs/report.md`、`outputs/trace.json`、`outputs/workspace.json`）。
