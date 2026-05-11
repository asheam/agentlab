# AgentLab (v0.3)

AgentLab 是一个面向学习与实验的轻量级 Python 多 Agent 框架。  
v0.1 跑通最小可用协作链路，v0.2 增强研究质量能力，v0.3 聚焦接口稳态重构（配置收敛、职责拆分、数据契约与稳定 CLI）。

## 1. 项目简介

核心流程：

用户任务 -> Supervisor 调度 -> Planner/Search/Reader/Critic/Writer 协作 -> 报告与追踪输出

适用场景：

- 学习多 Agent 架构与消息协议
- 快速验证 Agent 协作策略
- 低依赖、可测试的实验项目起点

## 2. 核心特性

- Agent 抽象：统一 `Agent` 接口与 `run()` 调用方式
- Message 协议：标准化 `task/response/tool_call/tool_result/error`
- LLM Adapter：`BaseModel` + `MockModel` + OpenAI-compatible 适配层
- Tool Registry：工具注册、查找、参数调用、错误处理
- Agent Runtime：按 `receiver` 路由消息并执行 Agent
- Supervisor-Worker 模式：固定顺序编排多 Agent 协作
- Blackboard：共享工作区沉淀中间产物
- Trace Recorder：记录调用事件、耗时、错误与工具执行信息
- Deep Research Demo：可一键运行完整研究流程并产出文件
- Critic 多模式：`--critic-mode auto|rule|llm`
- 维度化笔记：`notes.structured.dimensions`
- 对比矩阵：Writer 优先使用维度证据生成 `Comparison Matrix`
- v0.3 配置收敛：`SupervisorConfig` 统一配置入口
- v0.3 组合式组装：`build_default_tools` + `build_default_agents` + `build_default_supervisor`
- v0.3 稳定 CLI：`agentlab.cli.parse_args/main`
- v0.3 Workspace helpers：TypedDict + read/write helper（`workspace.research_workspace`）

## 3. 安装方式

前置要求：

- Python `3.11`
- 已安装 `uv`

安装依赖：

```bash
uv sync --dev
```

可选环境变量（使用 OpenAI-compatible 模型或 Tavily 时）：

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

Python API（v0.3 推荐）：

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

迁移说明：

- `build_default_supervisor` 旧 kwargs 调用仍可用，但会触发 `DeprecationWarning`
- 推荐改为 `build_default_supervisor(config=SupervisorConfig(...))`

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
  examples/
  tests/
  docs/
  outputs/
```

## 6. Deep Research Demo

默认运行：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别"
```

启用 OpenAI-compatible 模型：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别" --use-openai
```

真实检索 + 自动回退：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别" --search-mode real
```

严格真实检索（不回退）：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别" --search-mode real --no-search-fallback
```

Critic 模式：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别" --critic-mode auto
```

## 7. 输出文件说明

Demo 完成后生成：

- `outputs/report.md`：最终 Markdown 研究报告
- `outputs/trace.json`：执行轨迹与事件日志
- `outputs/workspace.json`：Blackboard 共享工作区快照

真实链路验收样例见：

- `docs/qa_acceptance.md`

## 8. Roadmap

v0.1（已完成）

- 跑通最小可用多 Agent 协作系统
- 保持轻依赖、可运行、可测试

v0.2（已完成）

- Critic `auto/rule/llm` 评审模式
- Reader 维度化结构化笔记
- Writer 维度优先对比矩阵
- Mock 检索维度语料增强
- 真实链路验收文档（rule/llm）

v0.3（已完成）

- `SupervisorConfig` 配置收敛 + 旧 kwargs 弃用过渡
- Supervisor 组装职责拆分（tools/agents/supervisor）
- `agentlab.cli` 稳定 CLI 入口与测试迁移
- Blackboard TypedDict 契约与访问助手（research workspace helpers）
- 接口稳态化回归（pytest/ruff/mypy + demo）

v0.4+（规划中）

- 更灵活的调度策略与失败重试
- 更细粒度的 trace 可视化
- 可替换的任务分解与笔记提炼策略
