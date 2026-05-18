# Tool Calling

工具调用通过 `ToolRegistry` 统一管理：

- `register(tool)`：注册工具
- `get(name)`：按名称查找
- `call(name, args)`：执行工具
- `list_tools()`：查看已注册工具

错误策略：

- 未注册工具：抛出明确异常
- 工具执行失败：包装为 `RuntimeError`

v0.1 内置：

- `calculator`
- `web_search`（mock）
