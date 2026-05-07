# AgentLab v0.1

AgentLab 是一个面向学习与实验的轻量级 Python 多 Agent 框架。v0.1 的目标是用最小可用架构跑通从任务拆解到报告生成的完整协作链路，而不是追求复杂的生产级能力。

## 1. 项目简介

AgentLab v0.1 聚焦多 Agent 协作机制验证，核心流程是：

用户任务 -> Supervisor 调度 -> Planner/Search/Reader/Critic/Writer 协作 -> 产出报告与追踪数据。

它适合用于：

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

## 3. 安装方式

前置要求：

- Python `3.11`
- 已安装 `uv`

安装依赖：

```bash
uv sync --dev
```

可选环境变量（使用 OpenAI 兼容模型时）：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

未配置 API Key 时，默认可使用 `MockModel` 跑通示例与测试。

## 4. 快速开始

```bash
uv sync --dev
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别"
```

质量检查命令：

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

## 5. 项目结构

```text
agentlab/
  pyproject.toml
  README.md
  .gitignore
  .python-version
  uv.lock
  src/
    agentlab/
      core/
      models/
      tools/
      workspace/
      tracing/
      multi_agent/
      agents/
  examples/
    01_single_agent.py
    02_tool_agent.py
    03_supervisor_worker.py
    04_deep_research.py
  tests/
  docs/
  outputs/
```

## 6. Deep Research Demo 运行方式

运行命令：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别"
```

启用 OpenAI-compatible API（例如 DeepSeek）：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别" --use-openai
```

自定义输出目录：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别" --output-dir outputs
```

预期流程：

1. Supervisor 接收用户主题
2. PlannerAgent 生成研究问题（写入 `plan`）
3. SearchAgent 检索信息（写入 `search_results`）
4. ReaderAgent 提炼要点（写入 `notes`）
5. CriticAgent 检查覆盖度（写入 `critique`）
6. WriterAgent 生成最终报告（写入 `report`）

## 7. 输出文件说明

Demo 运行后会在 `outputs/` 生成：

- `report.md`：最终 Markdown 研究报告
- `trace.json`：执行轨迹与事件日志
- `workspace.json`：Blackboard 共享工作区快照

这些文件用于回放过程、诊断问题、优化 Agent 协作策略。

## 8. Roadmap

v0.1（当前）

- 跑通最小可用多 Agent 协作系统
- 保持轻依赖、可运行、可测试

后续可选方向（v0.2+）

- 更灵活的调度策略与失败重试
- 更丰富的 Tool Calling 协议
- 更细粒度的 trace 可视化
- 可替换的记忆模块与任务分解策略

## 版本与约束

- Python 版本：`3.11`
- 依赖管理：`uv`
- 不引入重依赖（如 `torch`、`langchain`、`llamaindex` 等）

## License

Learning/experimental use. Add your preferred open-source license if needed.
