import pytest

from agentlab.tools.base import BaseTool
from agentlab.tools.calculator import CalculatorTool
from agentlab.tools.registry import ToolRegistry


class BrokenTool(BaseTool):
    name = "broken"
    description = "always fails"

    def run(self, **kwargs: object) -> object:
        raise ValueError("boom")


def test_register_get_and_list_tools() -> None:
    registry = ToolRegistry()
    calculator = CalculatorTool()

    registry.register(calculator)

    assert registry.get("calculator") is calculator
    assert "calculator" in registry.list_tools()


def test_call_tool_with_parameters() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    result = registry.call("calculator", {"expression": "1 + 1 * 2"})

    assert result == 3


def test_get_unknown_tool_raises_error() -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Tool 'missing' is not registered"):
        registry.get("missing")


def test_call_tool_exception_returns_clear_error() -> None:
    registry = ToolRegistry()
    registry.register(BrokenTool())

    with pytest.raises(RuntimeError, match="Failed to execute tool 'broken': boom"):
        registry.call("broken", {})


def test_register_duplicate_tool_name_raises_error() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    with pytest.raises(ValueError, match="Tool 'calculator' is already registered"):
        registry.register(CalculatorTool())
