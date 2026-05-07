from __future__ import annotations

from typing import Any

from agentlab.core.agent import Agent
from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message


class ReaderAgent(Agent):
    def __init__(self, model: Any = None) -> None:
        super().__init__(
            name="reader",
            role="reader",
            system_prompt="Read search results and extract concise research notes.",
            model=model,
        )

    def run(self, message: Message, context: RuntimeContext) -> Message:
        if context.blackboard is None:
            raise RuntimeError("blackboard is required for ReaderAgent")

        search_results = context.blackboard.read("search_results", [])
        notes = _build_notes(search_results)

        context.blackboard.write("notes", notes, author=self.name)
        return Message(
            sender=self.name,
            receiver=message.sender,
            content="Prepared summarized notes from search results.",
            type="response",
            metadata={"note_count": len(notes.get("key_points", []))},
        )


def _build_notes(search_results: Any) -> dict[str, Any]:
    key_points: list[str] = []
    references: list[str] = []

    if isinstance(search_results, list):
        for item in search_results:
            if not isinstance(item, dict):
                continue
            result = item.get("result")
            if isinstance(result, dict):
                for record in result.get("results", []):
                    if not isinstance(record, dict):
                        continue
                    title = str(record.get("title", ""))
                    snippet = str(record.get("snippet", ""))
                    source = str(record.get("source", ""))
                    if title or snippet:
                        key_points.append(f"{title}: {snippet}".strip())
                    if source:
                        references.append(source)

    dedup_points = list(dict.fromkeys(key_points))
    dedup_references = list(dict.fromkeys(references))
    return {
        "key_points": dedup_points,
        "references": dedup_references,
        "summary": "；".join(dedup_points[:6]) if dedup_points else "暂无可用检索结果。",
    }
