# Architecture

核心模块：

- `core/`：消息协议、Agent 抽象、运行时、事件
- `tools/`：工具抽象与注册中心
- `workspace/`：Blackboard 与 Artifact 存储
- `tracing/`：执行轨迹记录与导出
- `multi_agent/`：团队、调度器、Supervisor
- `agents/`：Planner/Search/Reader/Critic/Writer

主链路由 `Supervisor.run(topic)` 驱动，输出 `report.md`、`trace.json`、`workspace.json`。
