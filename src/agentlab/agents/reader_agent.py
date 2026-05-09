from __future__ import annotations

import re
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
            question = str(item.get("question", "")).strip()
            result = item.get("result")
            if not isinstance(result, dict):
                continue

            records = result.get("results", [])
            if not isinstance(records, list):
                continue

            dict_records = [record for record in records if isinstance(record, dict)]
            if not dict_records:
                continue

            mock_records = [
                record
                for record in dict_records
                if str(record.get("source", "")).startswith("mock://")
            ]
            if mock_records:
                selected = mock_records[:4]
            else:
                selected = [_pick_primary_record(dict_records)]

            for record in selected:
                title = str(record.get("title", ""))
                snippet = str(record.get("snippet", ""))
                source = str(record.get("source", ""))
                clean_title = _normalize_text(title, max_chars=90)
                clean_snippet = _normalize_text(snippet, max_chars=180)
                if clean_title or clean_snippet:
                    point = _compose_point(question, clean_title, clean_snippet)
                    key_points.append(point)
                if source:
                    references.append(source)

    dedup_points = list(dict.fromkeys(key_points))
    dedup_references = list(dict.fromkeys(references))
    return {
        "key_points": dedup_points[:16],
        "references": dedup_references[:16],
        "summary": "；".join(dedup_points[:4]) if dedup_points else "暂无可用检索结果。",
    }


_MULTI_SPACE = re.compile(r"\s+")


def _normalize_text(text: str, max_chars: int) -> str:
    compact = _MULTI_SPACE.sub(" ", text).strip()
    if not compact:
        return ""
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def _pick_primary_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        title = str(record.get("title", "")).lower()
        if "tavily answer" in title:
            return record
    return records[0]


def _compose_point(question: str, title: str, snippet: str) -> str:
    del question
    headline = snippet or title or "No summary"
    return headline
