from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from agentlab.agents.research_dimensions import (
    CONTENT_DIMENSION_KEYWORDS,
    DIMENSIONS,
    QUESTION_DIMENSION_KEYWORDS,
)
from agentlab.core.agent import Agent, ServiceName
from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message
from agentlab.models.base import BaseModel
from agentlab.workspace.research_workspace import (
    NotesPayload,
    SearchResultItem,
    read_search_results,
    write_notes,
)


@dataclass(frozen=True)
class ReaderStrategyInput:
    search_results: list[SearchResultItem]


class ReaderStrategy(Protocol):
    def build_notes(self, request: ReaderStrategyInput) -> NotesPayload:
        """Build structured notes from search results."""


class DefaultReaderStrategy:
    def build_notes(self, request: ReaderStrategyInput) -> NotesPayload:
        return _build_notes(request.search_results)


class ReaderAgent(Agent):
    def __init__(
        self,
        model: BaseModel | None = None,
        strategy: ReaderStrategy | None = None,
    ) -> None:
        super().__init__(
            name="reader",
            role="reader",
            system_prompt="Read search results and extract concise research notes.",
            model=model,
        )
        self.strategy = strategy or DefaultReaderStrategy()

    @property
    def required_services(self) -> set[ServiceName]:
        return {"blackboard"}

    def run(self, message: Message, context: RuntimeContext) -> Message:
        if context.blackboard is None:
            raise RuntimeError("blackboard is required for ReaderAgent")

        search_results = read_search_results(context.blackboard)
        notes = self.strategy.build_notes(ReaderStrategyInput(search_results=search_results))

        write_notes(context.blackboard, notes=notes, author=self.name)
        return Message(
            sender=self.name,
            receiver=message.sender,
            content="Prepared summarized notes from search results.",
            type="response",
            metadata={"note_count": len(notes.get("key_points", []))},
        )


_FRAMEWORKS = ("langgraph", "autogen", "crewai")
_FRAMEWORK_LABELS = {
    "langgraph": "LangGraph",
    "autogen": "AutoGen",
    "crewai": "CrewAI",
}


def _build_notes(search_results: Any) -> NotesPayload:
    key_points: list[str] = []
    references: list[str] = []
    structured_points: dict[str, list[str]] = {
        "langgraph": [],
        "autogen": [],
        "crewai": [],
        "common": [],
    }
    structured_references: dict[str, list[str]] = {
        "langgraph": [],
        "autogen": [],
        "crewai": [],
        "common": [],
    }
    structured_dimensions: dict[str, dict[str, list[str]]] = {
        dimension: {
            "langgraph": [],
            "autogen": [],
            "crewai": [],
            "common": [],
        }
        for dimension in DIMENSIONS
    }
    seen_dimension_points: dict[str, set[str]] = {
        "langgraph": set(),
        "autogen": set(),
        "crewai": set(),
        "common": set(),
    }

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
                framework = _detect_framework(
                    question=question,
                    title=clean_title,
                    snippet=clean_snippet,
                    source=source,
                )
                if clean_title or clean_snippet:
                    point = _compose_point(question, clean_title, clean_snippet, framework)
                    key_points.append(point)
                    structured_points[framework].append(point)
                    dimension = _detect_dimension(question, clean_title, clean_snippet)
                    normalized_point = _normalize_point_for_dedup(point)
                    if normalized_point not in seen_dimension_points[framework]:
                        structured_dimensions[dimension][framework].append(point)
                        seen_dimension_points[framework].add(normalized_point)
                if source:
                    references.append(source)
                    structured_references[framework].append(source)

    dedup_points = list(dict.fromkeys(key_points))
    dedup_references = list(dict.fromkeys(references))
    structured = _build_structured_notes(
        structured_points=structured_points,
        structured_references=structured_references,
        structured_dimensions=structured_dimensions,
    )

    notes: NotesPayload = {
        "key_points": dedup_points[:16],
        "references": dedup_references[:16],
        "summary": "；".join(dedup_points[:4]) if dedup_points else "暂无可用检索结果。",
        "structured": structured,
    }
    return notes


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


def _compose_point(_question: str, title: str, snippet: str, framework: str) -> str:
    headline = snippet or title or "No summary"
    if framework in _FRAMEWORK_LABELS:
        return f"{_FRAMEWORK_LABELS[framework]}: {headline}"
    return headline


def _detect_framework(question: str, title: str, snippet: str, source: str) -> str:
    corpus = f"{question}\n{title}\n{snippet}\n{source}".lower()
    hits = [name for name in _FRAMEWORKS if name in corpus]
    if len(hits) == 1:
        return hits[0]
    return "common"


def _detect_dimension(question: str, title: str, snippet: str) -> str:
    by_question = _detect_dimension_from_question(question)
    if by_question is not None:
        return by_question
    return _detect_dimension_from_text(question, title, snippet)


def _detect_dimension_from_question(question: str) -> str | None:
    lowered = question.lower()
    for dimension in DIMENSIONS:
        keywords = QUESTION_DIMENSION_KEYWORDS.get(dimension, ())
        if any(keyword.lower() in lowered for keyword in keywords):
            return dimension
    return None


def _detect_dimension_from_text(question: str, title: str, snippet: str) -> str:
    corpus = f"{question}\n{title}\n{snippet}".lower()
    best_dimension = "core_paradigm"
    best_score = -1
    for dimension, keywords in CONTENT_DIMENSION_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in corpus)
        if score > best_score:
            best_score = score
            best_dimension = dimension
    return best_dimension


def _normalize_point_for_dedup(text: str) -> str:
    lowered = text.lower()
    normalized = _MULTI_SPACE.sub(" ", lowered).strip()
    return normalized


def _build_structured_notes(
    structured_points: dict[str, list[str]],
    structured_references: dict[str, list[str]],
    structured_dimensions: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("langgraph", "autogen", "crewai", "common"):
        points = [point for point in structured_points.get(key, []) if point]
        refs = [ref for ref in structured_references.get(key, []) if ref]
        payload[key] = {
            "points": list(dict.fromkeys(points))[:8],
            "references": list(dict.fromkeys(refs))[:8],
        }

    dimension_payload: dict[str, dict[str, list[str]]] = {}
    for dimension in DIMENSIONS:
        node = structured_dimensions.get(dimension, {})
        dimension_payload[dimension] = {}
        for framework in ("langgraph", "autogen", "crewai", "common"):
            raw_points = node.get(framework, [])
            points = [point for point in raw_points if point]
            dimension_payload[dimension][framework] = list(dict.fromkeys(points))[:6]

    payload["dimensions"] = dimension_payload
    return payload
