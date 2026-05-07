# Agent Loop

单次 Agent 执行循环：

1. Runtime 接收 Message
2. 根据 `receiver` 路由到对应 Agent
3. Agent 读取 Context（tools/blackboard/tracing/artifacts）
4. Agent 产出 Message
5. Runtime 记录 Event

该循环是单步同步模型，便于学习与调试。
