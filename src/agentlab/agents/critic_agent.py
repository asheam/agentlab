from __future__ import annotations

from typing import Any

from agentlab.core.agent import Agent
from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message


class CriticAgent(Agent):
    def __init__(self, model: Any = None) -> None:
        super().__init__(
            name="critic",
            role="critic",
            system_prompt="Evaluate note quality and identify coverage gaps.",
            model=model,
        )

    def run(self, message: Message, context: RuntimeContext) -> Message:
        if context.blackboard is None:
            raise RuntimeError("blackboard is required for CriticAgent")

        notes = context.blackboard.read("notes", {})
        critique = _build_critique(notes)

        context.blackboard.write("critique", critique, author=self.name)
        return Message(
            sender=self.name,
            receiver=message.sender,
            content="Generated critique for current notes.",
            type="response",
            metadata=critique,
        )


def _build_critique(notes: Any) -> dict[str, Any]:
    summary = ""
    key_points: list[str] = []
    if isinstance(notes, dict):
        summary = str(notes.get("summary", ""))
        raw_points = notes.get("key_points", [])
        if isinstance(raw_points, list):
            key_points = [str(item) for item in raw_points]

    corpus = " ".join(key_points + [summary]).lower()
    required = ["langgraph", "autogen", "crewai"]
    missing = [name for name in required if name not in corpus]

    strengths = [
        "已形成结构化要点",
        "包含可追溯的参考来源",
    ]
    gaps = [f"缺少对 {name} 的充分覆盖" for name in missing] if missing else ["覆盖完整度良好"]
    verdict = "need_improvement" if missing else "acceptable"

    return {
        "strengths": strengths,
        "gaps": gaps,
        "verdict": verdict,
    }
