from __future__ import annotations

from agentlab.tools.calculator import CalculatorTool
from agentlab.tools.registry import ToolRegistry


if __name__ == "__main__":
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    expression = "1 + 1 * 3"
    result = registry.call("calculator", {"expression": expression})

    print(f"expression: {expression}")
    print(f"result: {result}")
