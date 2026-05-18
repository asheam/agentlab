# AgentLab (v0.3)

AgentLab 是一个面向学习与实验的轻量级 Python 多 Agent 框架。

v0.1 跑通最小可用协作链路，v0.2 增强研究质量能力，v0.3 聚焦接口稳态重构（配置收敛、组装拆分、Workspace 契约与稳定 CLI）。

## 1. 项目简介

核心流程：

用户任务 -> Supervisor 调度 -> Planner/Search/Reader/Critic/Writer 协作 -> 报告与追踪输出

适用场景：

- 学习多 Agent 架构与消息协作
- 快速验证 Agent 协作策略
- 作为低依赖、可测试的研究框架起点

## 2. 核心特性

- Agent 抽象：统一 `Agent.run()` 执行接口
- Message 协议：标准化 `task/response/tool_call/tool_result/error`
- LLM Adapter：`BaseModel` + `MockModel` + OpenAI-compatible 适配层
- Tool Registry：注册、查找、调用、错误处理
- Agent Runtime：按 `receiver` 路由消息并执行 Agent
- Supervisor-Worker：固定顺序的多 Agent 协作
- Blackboard：共享工作区沉淀中间产物
- Trace Recorder：记录执行事件、耗时、错误、工具调用
- Critic 多模式：`auto | rule | llm`
- 维度化笔记：`notes.structured.dimensions`
- 稳定 CLI：`agentlab.cli.parse_args/main`
- 配置收敛：`SupervisorConfig`

## 3. 安装方式

前置：

- Python `3.11`
- `uv`

安装依赖：

```bash
uv sync --dev
```

可选环境变量（启用 OpenAI-compatible 或 Tavily 时）：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `TAVILY_API_KEY`

## 4. 快速开始

```bash
uv sync --dev
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别"
```

质量检查：

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

Python API（推荐 v0.3 方式）：

```python
from agentlab.multi_agent.supervisor import SupervisorConfig, build_default_supervisor

config = SupervisorConfig(
    output_dir="outputs",
    search_mode="mock",
    critic_mode="auto",
)
supervisor = build_default_supervisor(config=config)
outputs = supervisor.run("研究 LangGraph、AutoGen、CrewAI 的区别")
```

## 5. 项目结构

```text
agentlab/
  src/agentlab/
    core/
    models/
    tools/
    workspace/
    tracing/
    multi_agent/
    agents/
    cli.py
  examples/
  tests/
  docs/
  outputs/
```

## 6. Demo 运行

默认 Deep Research：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别"
```

启用 OpenAI-compatible 模型：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别" --use-openai
```

真实检索（允许回退）：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别" --search-mode real
```

自定义 Planner/Writer 策略示例：

```bash
uv run python examples/05_custom_strategies.py "研究 LangGraph、AutoGen、CrewAI 的区别"
```

## 7. 输出文件说明

运行完成后输出：

- `outputs/report.md`：最终研究报告
- `outputs/trace.json`：执行轨迹明细
- `outputs/workspace.json`：Blackboard 快照
- `outputs/run_summary.json`：执行摘要统计

## 8. 自定义策略接口

v0.4 基础能力已就绪：可通过 `SupervisorConfig` 注入可替换策略。

```python
from agentlab.multi_agent.supervisor import SupervisorConfig

config = SupervisorConfig(
    planner_strategy=MyPlannerStrategy(),
    writer_strategy=MyWriterStrategy(),
)
```

策略签名：

- Planner: `build_plan(topic: str, model: BaseModel | None) -> list[str]`
- Writer: `build_report(request: WriterStrategyInput) -> str`

完整示例见：`examples/05_custom_strategies.py` 与 `docs/custom_strategies.md`。

## 9. Roadmap

v0.1（已完成）：

- 最小可用多 Agent 协作框架

v0.2（已完成）：

- Critic 多模式、维度化笔记、对比矩阵增强、真实链路验收

v0.3（已完成）：

- `SupervisorConfig` 配置收敛
- 组装职责拆分（tools / agents / supervisor）
- 稳定 CLI 与测试迁移
- Workspace TypedDict + helper

v0.4（进行中）：

- 可替换 Planner/Writer 策略能力（已实现基础接口）
- 文档、示例与更多扩展点固化
