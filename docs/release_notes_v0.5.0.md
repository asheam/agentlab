# AgentLab v0.5.0 Release Notes (Draft)

发布日期：2026-05-19

## 版本目标

v0.5.0 聚焦“接口稳态”：统一配置入口、统一 CLI 入口、收敛 Blackboard 访问契约，保持业务行为与输出语义不变。

## 主要改动

1. Blackboard typed helper 全链路收口
- 在 `workspace/research_workspace.py` 新增 notes/critique/search_result typed accessor。
- Writer/Critic/Reader 默认链路改为优先使用 helper，减少分散的 `isinstance` 防御分支。
- 保留 `Blackboard.read/write` 通用 API，兼容已有扩展点。

2. SupervisorConfig 成为主入口
- `build_default_supervisor` 继续兼容 legacy kwargs（带弃用告警），但主调用路径统一为 `config=SupervisorConfig(...)`。
- 默认示例与包内运行入口均改为显式 `SupervisorConfig`。

3. CLI 稳定入口统一
- `pyproject.toml` 的 `[project.scripts]` 入口更新为 `agentlab.cli:entrypoint`。
- 新增 CLI 入口冒烟测试（`--help` 退出码与 entrypoint 退出码透传）。

## 推荐运行方式

```bash
uv run agentlab "研究 LangGraph、AutoGen、CrewAI 的区别"
```

```bash
uv run agentlab --config docs/agentlab.sample.yaml --print-effective-config
```

## 兼容性与迁移

- 兼容：`examples/04_deep_research.py` 仍可运行（作为薄包装入口）。
- 兼容：`build_default_supervisor(...)` 的 legacy kwargs 仍可用，但建议迁移到 `SupervisorConfig`。
- 配置优先级不变：`CLI 参数 > 配置文件 > 默认值`。

## 验证状态（草案）

- `pytest` 通过
- `ruff check .` 通过
- `mypy src` 通过
