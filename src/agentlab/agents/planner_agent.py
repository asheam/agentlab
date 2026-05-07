from __future__ import annotations

from typing import Any

from agentlab.core.agent import Agent
from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message


class PlannerAgent(Agent):
    def __init__(self, model: Any = None) -> None:
        super().__init__(
            name="planner",
            role="planner",
            system_prompt="Break down a research topic into focused questions.",
            model=model,
        )

    def run(self, message: Message, context: RuntimeContext) -> Message:
        topic = message.content.strip()
        plan = _build_plan(topic)

        if context.blackboard is not None:
            context.blackboard.write("plan", plan, author=self.name)

        return Message(
            sender=self.name,
            receiver=message.sender,
            content="Generated research plan with focused questions.",
            type="response",
            metadata={"plan": plan},
        )


def _build_plan(topic: str) -> list[str]:
    normalized = topic.lower()
    if "langgraph" in normalized or "autogen" in normalized or "crewai" in normalized:
        return [
            "它们各自的核心设计理念和目标场景是什么？",
            "任务编排方式（图、对话、角色分工）有哪些关键差异？",
            "工具调用、状态管理和可观测性能力分别如何？",
            "在学习成本、扩展性和工程落地上各自的优缺点是什么？",
            "针对中小型团队做技术选型时应如何决策？",
        ]

    return [
        f"{topic} 的核心概念和边界是什么？",
        f"{topic} 的主流实现路径有哪些？",
        f"{topic} 的关键评估指标是什么？",
        f"{topic} 在实际落地中的常见风险是什么？",
    ]
