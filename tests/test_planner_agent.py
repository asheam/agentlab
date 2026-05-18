from __future__ import annotations

from agentlab.agents.planner_agent import PlannerAgent
from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message
from agentlab.models.base import BaseModel
from agentlab.workspace.blackboard import Blackboard


class _StaticPlannerStrategy:
    def build_plan(self, topic: str, model: BaseModel | None) -> list[str]:
        del topic, model
        return ["Q1 custom", "Q2 custom"]


def test_planner_agent_uses_custom_strategy_and_writes_plan() -> None:
    board = Blackboard()
    context = RuntimeContext(blackboard=board)
    agent = PlannerAgent(strategy=_StaticPlannerStrategy())

    response = agent.run(
        Message(sender="supervisor", receiver="planner", content="topic", type="task"),
        context,
    )

    assert response.type == "response"
    assert response.metadata["plan"] == ["Q1 custom", "Q2 custom"]
    assert board.read("plan") == ["Q1 custom", "Q2 custom"]
