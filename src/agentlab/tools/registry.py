from __future__ import annotations

from typing import Any
from collections.abc import Mapping

from agentlab.tools.base import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError("Tool must define a non-empty name")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' is not registered")
        return tool

    def call(self, name: str, args: Mapping[str, Any] | None = None) -> Any:
        tool = self.get(name)
        kwargs = dict(args or {})
        try:
            return tool.run(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to execute tool '{name}': {exc}") from exc

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())
