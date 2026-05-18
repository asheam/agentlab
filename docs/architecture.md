# Architecture

核心模块：

- `core/`：消息协议、Agent 抽象、运行时、事件
- `tools/`：工具抽象与注册中心
- `workspace/`：Blackboard 与 Artifact 存储
- `tracing/`：执行轨迹记录与导出
- `multi_agent/`：团队、调度器、Supervisor
- `agents/`：Planner/Search/Reader/Critic/Writer

主链路由 `Supervisor.run(topic)` 驱动，输出 `report.md`、`trace.json`、`workspace.json`、`run_summary.json`。

扩展点：

- 通过 `SupervisorConfig.planner_strategy` 注入自定义 Planner 任务拆解策略
- 通过 `SupervisorConfig.writer_strategy` 注入自定义 Writer 报告生成策略
- 无需改动 Runtime 或调度器即可替换研究链路中的关键行为
