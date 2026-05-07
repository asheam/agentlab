# AgentLab v0.1 设计说明

本版本目标是跑通最小可用多 Agent 协作链路：

1. 用户输入主题
2. Supervisor 固定顺序调度 5 个 Agent
3. Blackboard 沉淀中间产物
4. Writer 生成报告
5. Trace 与 Workspace 导出到 outputs/

不追求生产级特性，只强调结构清晰、可运行、可测试。
