# Multi-Agent 模式

v0.1 使用固定顺序的 Supervisor-Worker：

1. PlannerAgent
2. SearchAgent
3. ReaderAgent
4. CriticAgent
5. WriterAgent

优势是实现简单、行为可预期；后续可扩展动态调度或重试策略。
