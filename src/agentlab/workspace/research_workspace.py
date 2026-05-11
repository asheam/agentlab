from __future__ import annotations

from typing import Any, TypedDict, cast

from agentlab.workspace.blackboard import Blackboard, BlackboardEntry


class SearchResultItem(TypedDict, total=False):
    question: str
    result: dict[str, Any]
    error: str


class NotesPayload(TypedDict, total=False):
    key_points: list[str]
    references: list[str]
    summary: str
    structured: dict[str, Any]


class CritiquePayload(TypedDict, total=False):
    strengths: list[str]
    gaps: list[str]
    verdict: str
    scores: dict[str, Any]
    stats: dict[str, Any]
    recommendations: list[str]
    assessment_mode: str
    model_fallback_reason: str
    rule_backup: dict[str, Any]


class BlackboardEntity(TypedDict, total=False):
    plan: list[str]
    search_results: list[SearchResultItem]
    notes: NotesPayload
    critique: CritiquePayload
    report: str


def read_plan(board: Blackboard) -> list[str]:
    raw = board.read("plan", [])
    if not isinstance(raw, list):
        return []
    plan: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text:
            plan.append(text)
    return plan


def write_plan(board: Blackboard, plan: list[str], author: str) -> BlackboardEntry:
    return board.write("plan", list(plan), author=author)


def read_search_results(board: Blackboard) -> list[SearchResultItem]:
    raw = board.read("search_results", [])
    if not isinstance(raw, list):
        return []

    items: list[SearchResultItem] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        normalized: SearchResultItem = {}
        question = item.get("question")
        if question is not None:
            normalized["question"] = str(question)
        result = item.get("result")
        if isinstance(result, dict):
            normalized["result"] = cast(dict[str, Any], result)
        error = item.get("error")
        if error is not None:
            normalized["error"] = str(error)
        items.append(normalized)
    return items


def write_search_results(
    board: Blackboard,
    search_results: list[SearchResultItem],
    author: str,
) -> BlackboardEntry:
    return board.write("search_results", list(search_results), author=author)


def read_notes(board: Blackboard) -> NotesPayload:
    raw = board.read("notes", {})
    if not isinstance(raw, dict):
        return NotesPayload(key_points=[], references=[], summary="", structured={})

    key_points = _as_string_list(raw.get("key_points", []))
    references = _as_string_list(raw.get("references", []))
    summary = str(raw.get("summary", ""))
    structured = raw.get("structured", {})
    if not isinstance(structured, dict):
        structured = {}

    return NotesPayload(
        key_points=key_points,
        references=references,
        summary=summary,
        structured=cast(dict[str, Any], structured),
    )


def write_notes(board: Blackboard, notes: NotesPayload, author: str) -> BlackboardEntry:
    return board.write("notes", dict(notes), author=author)


def read_critique(board: Blackboard) -> CritiquePayload:
    raw = board.read("critique", {})
    if not isinstance(raw, dict):
        return CritiquePayload()

    payload: dict[str, Any] = {}
    for key in ("verdict", "assessment_mode", "model_fallback_reason"):
        value = raw.get(key)
        if isinstance(value, str):
            payload[key] = value

    for key in ("strengths", "gaps", "recommendations"):
        critique_list = _as_string_list(raw.get(key, []))
        if critique_list:
            payload[key] = critique_list

    for key in ("scores", "stats", "rule_backup"):
        value = raw.get(key)
        if isinstance(value, dict):
            payload[key] = cast(dict[str, Any], value)

    return cast(CritiquePayload, payload)


def write_critique(board: Blackboard, critique: CritiquePayload, author: str) -> BlackboardEntry:
    return board.write("critique", dict(critique), author=author)


def read_report(board: Blackboard) -> str:
    raw = board.read("report", "")
    if isinstance(raw, str):
        return raw
    return str(raw)


def write_report(board: Blackboard, report: str, author: str) -> BlackboardEntry:
    return board.write("report", report, author=author)


def _as_string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []

    items: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text:
            items.append(text)
    return items
