from __future__ import annotations

import pytest

from agentlab.agents.search_agent import SearchAgent, SearchStrategyInput, SearchStrategyOutput
from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message
from agentlab.tools.registry import ToolRegistry
from agentlab.workspace.blackboard import Blackboard


class _StaticSearchStrategy:
    def __init__(self, output: SearchStrategyOutput) -> None:
        self.output = output
        self.last_request: SearchStrategyInput | None = None

    def collect(self, request: SearchStrategyInput) -> SearchStrategyOutput:
        self.last_request = request
        return self.output


def test_search_agent_uses_custom_strategy_and_writes_results() -> None:
    board = Blackboard()
    board.write("plan", ["Q1"], author="planner")
    strategy = _StaticSearchStrategy(
        SearchStrategyOutput(
            search_results=[{"question": "Q1", "result": {"results": [{"title": "t", "source": "mock://x"}]}}],
            tool_errors=[],
        )
    )
    agent = SearchAgent(strategy=strategy)
    context = RuntimeContext(blackboard=board, tool_registry=ToolRegistry())

    response = agent.run(
        Message(
            sender="supervisor",
            receiver="searcher",
            content="search",
            type="task",
            metadata={"topic": "topic"},
        ),
        context,
    )

    assert response.type == "response"
    assert response.metadata["count"] == 1
    assert board.read("search_results")[0]["question"] == "Q1"
    assert strategy.last_request is not None
    assert strategy.last_request.topic == "topic"


def test_search_agent_raises_when_custom_strategy_reports_error_and_strict_mode() -> None:
    board = Blackboard()
    board.write("plan", ["Q1"], author="planner")
    strategy = _StaticSearchStrategy(
        SearchStrategyOutput(
            search_results=[{"question": "Q1", "error": "simulated error"}],
            tool_errors=["simulated error"],
        )
    )
    agent = SearchAgent(strategy=strategy, fail_on_tool_error=True)
    context = RuntimeContext(blackboard=board, tool_registry=ToolRegistry())

    with pytest.raises(RuntimeError, match="simulated error"):
        agent.run(
            Message(sender="supervisor", receiver="searcher", content="search", type="task"),
            context,
        )
