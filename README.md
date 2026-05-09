# AgentLab（v0.1 基线 + v0.2 候选增强）

AgentLab 是一个面向学习与实验的轻量级 Python 多 Agent 框架。  
v0.1 目标是跑通最小可用协作链路；当前仓库在此基础上加入了 v0.2 候选增强（维度化笔记、对比矩阵、Critic 多模式、真实链路验收）。

## 1. 项目简介

AgentLab 聚焦多 Agent 协作机制验证，核心流程是：

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
- Critic 多模式：`--critic-mode auto|rule|llm`（支持 LLM 评审与规则回退）
- 维度化笔记：`notes.structured.dimensions`（`core_paradigm/coordination_style/state_memory/best_fit/trade_off`）
- 对比矩阵：Writer 优先使用维度证据生成 `Comparison Matrix`
- Mock 维度语料：离线模式按维度返回不同片段，减少模板化输出

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
- `TAVILY_API_KEY`（使用 Tavily real search 时）

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

启用真实联网检索（失败会自动回退 mock）：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别" --search-mode real
```

`real` 模式会按 provider 顺序尝试检索（默认：`duckduckgo,wikipedia,tavily`），并在报告的 `Search Mode Summary` 中显示命中与错误统计。

显式优先使用 Tavily：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别" --search-mode real --search-providers tavily,duckduckgo,wikipedia
```

启用真实联网检索且禁止回退（用于严格验证 real 路径）：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别" --search-mode real --no-search-fallback
```

Critic 评审模式（`auto`/`rule`/`llm`）：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别" --critic-mode auto
```

- `auto`：有模型时优先用 LLM 评审，失败自动回退规则评审
- `rule`：只使用规则评审（稳定、可离线）
- `llm`：强制尝试 LLM 评审，无模型或失败时标记为 `rule_fallback`

真实检索 + LLM Critic 示例：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别" --search-mode real --critic-mode auto --use-openai
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

说明：

- `report.md` 默认保留摘要、对比矩阵、评审结果和引用。
- `workspace.json` 中 `notes.structured.dimensions` 可用于审查维度覆盖情况。
- `critique` 中包含 `dimension_coverage`、`dimensions_covered`、`dimensions_missing` 等质量信号。
- 调试和复盘请优先查看 `trace.json` 与 `workspace.json`（包含更完整过程数据）。

这些文件用于回放过程、诊断问题、优化 Agent 协作策略。

真实链路验收样例（rule/llm 两条链路）见：

- `docs/qa_acceptance.md`

## 8. Roadmap

v0.1（已完成）

- 跑通最小可用多 Agent 协作系统
- 保持轻依赖、可运行、可测试

v0.2（当前候选）

- Critic `auto/rule/llm` 评审模式
- Reader 维度化结构化笔记
- Writer 维度优先对比矩阵
- Mock 检索维度语料增强
- 真实链路验收文档（rule/llm）

后续方向（v0.3+）

- 更灵活的调度策略与失败重试
- 更细粒度的 trace 可视化
- 可替换的任务分解与笔记提炼策略

## 版本与约束

- Python 版本：`3.11`
- 依赖管理：`uv`
- 不引入重依赖（如 `torch`、`langchain`、`llamaindex` 等）

## License

Learning/experimental use. Add your preferred open-source license if needed.
