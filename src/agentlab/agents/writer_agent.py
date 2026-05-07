from __future__ import annotations

from typing import Any

from agentlab.core.agent import Agent
from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message


class WriterAgent(Agent):
    def __init__(self, model: Any = None) -> None:
        super().__init__(
            name="writer",
            role="writer",
            system_prompt="Write final markdown report from shared workspace.",
            model=model,
        )

    def run(self, message: Message, context: RuntimeContext) -> Message:
        if context.blackboard is None:
            raise RuntimeError("blackboard is required for WriterAgent")

        topic = message.metadata.get("topic") if isinstance(message.metadata, dict) else None
        if not isinstance(topic, str) or not topic.strip():
            topic = message.content.strip() or "Untitled Topic"

        plan = context.blackboard.read("plan", [])
        search_results = context.blackboard.read("search_results", [])
        notes = context.blackboard.read("notes", {})
        critique = context.blackboard.read("critique", {})

        report = _build_report(topic, plan, search_results, notes, critique)
        context.blackboard.write("report", report, author=self.name)

        if context.artifacts is not None:
            context.artifacts.save_text("report.md", report)

        return Message(
            sender=self.name,
            receiver=message.sender,
            content=report,
            type="response",
            metadata={"topic": topic},
        )


def _build_report(
    topic: str,
    plan: Any,
    search_results: Any,
    notes: Any,
    critique: Any,
) -> str:
    lines: list[str] = []
    lines.append("# Deep Research Report")
    lines.append("")
    lines.append(f"## Topic\n{topic}")
    lines.append("")

    lines.append("## Research Questions")
    if isinstance(plan, list) and plan:
        for item in plan:
            lines.append(f"- {item}")
    else:
        lines.append("- 无可用研究问题")
    lines.append("")

    lines.append("## Key Findings")
    key_points = notes.get("key_points", []) if isinstance(notes, dict) else []
    if isinstance(key_points, list) and key_points:
        for point in key_points[:12]:
            lines.append(f"- {point}")
    else:
        lines.append("- 暂无检索要点")
    lines.append("")

    lines.append("## Framework Snapshot")
    for framework, evidence in _framework_snapshot(key_points):
        lines.append(f"- {framework}: {evidence}")
    lines.append("")

    lines.append("## Critique")
    if isinstance(critique, dict):
        for strength in critique.get("strengths", []):
            lines.append(f"- Strength: {strength}")
        for gap in critique.get("gaps", []):
            lines.append(f"- Gap: {gap}")
        lines.append(f"- Verdict: {critique.get('verdict', 'unknown')}")
    else:
        lines.append("- 无评审信息")
    lines.append("")

    lines.append("## References")
    references = notes.get("references", []) if isinstance(notes, dict) else []
    if isinstance(references, list) and references:
        for ref in references:
            lines.append(f"- {ref}")
    else:
        lines.append("- 无引用")
    lines.append("")

    lines.append("## Raw Search Snapshot")
    if isinstance(search_results, list) and search_results:
        lines.append("```json")
        lines.append(_safe_json(search_results[:3]))
        lines.append("```")
    else:
        lines.append("无检索快照")

    return "\n".join(lines).strip() + "\n"


def _safe_json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)


def _framework_snapshot(key_points: Any) -> list[tuple[str, str]]:
    if not isinstance(key_points, list):
        return [("General", "No framework-specific findings.")]

    corpus = " ".join(str(item) for item in key_points).lower()
    snapshots: list[tuple[str, str]] = []
    if "langgraph" in corpus:
        snapshots.append(
            ("LangGraph", "Graph/state-machine orchestration is highlighted for complex flows.")
        )
    if "autogen" in corpus:
        snapshots.append(
            ("AutoGen", "Conversation-centric agent collaboration appears as a core strength.")
        )
    if "crewai" in corpus:
        snapshots.append(
            ("CrewAI", "Role/task-driven execution is emphasized for practical team workflows.")
        )
    if not snapshots:
        snapshots.append(("General", "No framework-specific findings."))
    return snapshots
