from __future__ import annotations

from typing import Any, Literal, TypedDict, cast

from agentlab.workspace.blackboard import Blackboard, BlackboardEntry

FrameworkName = Literal["langgraph", "autogen", "crewai", "common"]


class SearchSourceHits(TypedDict, total=False):
    duckduckgo: int
    wikipedia: int
    tavily: int


class SearchProviderErrors(TypedDict, total=False):
    duckduckgo: int
    wikipedia: int
    tavily: int


class SearchToolResult(TypedDict, total=False):
    mode: str
    fallback_used: bool
    fallback_reason: str
    source_hits: SearchSourceHits
    provider_errors: SearchProviderErrors
    real_issues: list[str] | str
    results: list[dict[str, Any]]


class SearchResultItem(TypedDict, total=False):
    question: str
    result: SearchToolResult
    error: str


class StructuredFrameworkNode(TypedDict, total=False):
    points: list[str]
    references: list[str]


class StructuredDimensionNode(TypedDict, total=False):
    langgraph: list[str]
    autogen: list[str]
    crewai: list[str]
    common: list[str]


class StructuredNotesPayload(TypedDict, total=False):
    langgraph: StructuredFrameworkNode
    autogen: StructuredFrameworkNode
    crewai: StructuredFrameworkNode
    common: StructuredFrameworkNode
    dimensions: dict[str, StructuredDimensionNode]


class NotesPayload(TypedDict, total=False):
    key_points: list[str]
    references: list[str]
    summary: str
    structured: StructuredNotesPayload


class CritiqueScores(TypedDict, total=False):
    overall: Any
    coverage: Any
    evidence: Any
    specificity: Any
    balance: Any
    reasoning_depth: Any
    dimension_coverage: Any


class CritiqueStats(TypedDict, total=False):
    key_points_count: Any
    references_count: Any
    missing_frameworks: list[str]
    comparative_points_count: Any
    limitation_points_count: Any
    actionable_points_count: Any
    source_domains_count: Any
    source_domains: list[str]
    dimensions_covered_count: Any
    dimensions_covered: list[str]
    dimensions_missing: list[str]


class CritiquePayload(TypedDict, total=False):
    strengths: list[str]
    gaps: list[str]
    verdict: str
    scores: CritiqueScores
    stats: CritiqueStats
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
            normalized["result"] = cast(SearchToolResult, result)
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
        structured=cast(StructuredNotesPayload, structured),
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


def notes_key_points(notes: NotesPayload) -> list[str]:
    return _as_string_list(notes.get("key_points", []))


def notes_references(notes: NotesPayload) -> list[str]:
    return _as_string_list(notes.get("references", []))


def notes_summary(notes: NotesPayload) -> str:
    value = notes.get("summary", "")
    if isinstance(value, str):
        return value
    return str(value)


def notes_structured(notes: NotesPayload) -> StructuredNotesPayload:
    raw = notes.get("structured", {})
    if not isinstance(raw, dict):
        return StructuredNotesPayload()

    payload: StructuredNotesPayload = {}
    for framework in _FRAMEWORKS:
        framework_node = _extract_framework_node(raw.get(framework))
        if framework_node:
            payload[framework] = framework_node

    raw_dimensions = raw.get("dimensions")
    if isinstance(raw_dimensions, dict):
        dimensions_payload: dict[str, StructuredDimensionNode] = {}
        for dimension_key, raw_node in raw_dimensions.items():
            if not isinstance(dimension_key, str):
                continue
            dimension_node = _extract_dimension_node(raw_node)
            if dimension_node:
                dimensions_payload[dimension_key] = dimension_node
        if dimensions_payload:
            payload["dimensions"] = dimensions_payload

    return payload


def notes_framework_points(notes: NotesPayload, framework: FrameworkName) -> list[str]:
    structured = notes_structured(notes)
    node = structured.get(framework)
    if not node:
        return []
    return _as_string_list(node.get("points", []))


def notes_dimension_points(
    notes: NotesPayload,
    dimension: str,
    framework: FrameworkName,
) -> list[str]:
    structured = notes_structured(notes)
    dimensions = structured.get("dimensions", {})
    node = dimensions.get(dimension)
    if not node:
        return []
    return _as_string_list(node.get(framework, []))


def critique_verdict(critique: CritiquePayload, default: str = "unknown") -> str:
    value = critique.get("verdict")
    if isinstance(value, str) and value.strip():
        return value
    return default


def critique_assessment_mode(critique: CritiquePayload) -> str | None:
    value = critique.get("assessment_mode")
    if isinstance(value, str) and value.strip():
        return value
    return None


def critique_strengths(critique: CritiquePayload) -> list[str]:
    return _as_string_list(critique.get("strengths", []))


def critique_gaps(critique: CritiquePayload) -> list[str]:
    return _as_string_list(critique.get("gaps", []))


def critique_recommendations(critique: CritiquePayload) -> list[str]:
    return _as_string_list(critique.get("recommendations", []))


def critique_scores(critique: CritiquePayload) -> CritiqueScores:
    raw = critique.get("scores")
    if not isinstance(raw, dict):
        return CritiqueScores()
    return cast(CritiqueScores, raw)


def critique_stats(critique: CritiquePayload) -> CritiqueStats:
    raw = critique.get("stats")
    if not isinstance(raw, dict):
        return CritiqueStats()
    return cast(CritiqueStats, raw)


def search_result_error(item: SearchResultItem) -> str | None:
    value = item.get("error")
    if isinstance(value, str) and value.strip():
        return value
    return None


def search_result_payload(item: SearchResultItem) -> SearchToolResult | None:
    value = item.get("result")
    if isinstance(value, dict):
        return cast(SearchToolResult, value)
    return None


def search_result_mode(payload: SearchToolResult | None) -> str:
    if payload is None:
        return ""
    mode = payload.get("mode")
    if isinstance(mode, str):
        return mode
    return ""


def search_result_source_hits(payload: SearchToolResult | None) -> SearchSourceHits:
    if payload is None:
        return SearchSourceHits()

    raw = payload.get("source_hits")
    if not isinstance(raw, dict):
        return SearchSourceHits()

    return SearchSourceHits(
        duckduckgo=_as_int(raw.get("duckduckgo")),
        wikipedia=_as_int(raw.get("wikipedia")),
        tavily=_as_int(raw.get("tavily")),
    )


def search_result_provider_errors(payload: SearchToolResult | None) -> SearchProviderErrors:
    if payload is None:
        return SearchProviderErrors()

    raw = payload.get("provider_errors")
    if not isinstance(raw, dict):
        return SearchProviderErrors()

    return SearchProviderErrors(
        duckduckgo=_as_int(raw.get("duckduckgo")),
        wikipedia=_as_int(raw.get("wikipedia")),
        tavily=_as_int(raw.get("tavily")),
    )


def search_result_issues(payload: SearchToolResult | None) -> list[str]:
    if payload is None:
        return []
    raw_issues = payload.get("real_issues")
    if isinstance(raw_issues, str):
        issue = raw_issues.strip()
        return [issue] if issue else []
    return _as_string_list(raw_issues)


def search_result_fallback_used(payload: SearchToolResult | None) -> bool:
    if payload is None:
        return False
    return payload.get("fallback_used") is True


def search_result_fallback_reason(payload: SearchToolResult | None) -> str | None:
    if payload is None:
        return None
    raw = payload.get("fallback_reason")
    if isinstance(raw, str) and raw.strip():
        return raw
    return None


def _extract_framework_node(raw: Any) -> StructuredFrameworkNode:
    if not isinstance(raw, dict):
        return StructuredFrameworkNode()
    points = _as_string_list(raw.get("points", []))
    references = _as_string_list(raw.get("references", []))
    node: StructuredFrameworkNode = {}
    if points:
        node["points"] = points
    if references:
        node["references"] = references
    return node


def _extract_dimension_node(raw: Any) -> StructuredDimensionNode:
    if not isinstance(raw, dict):
        return StructuredDimensionNode()
    node: StructuredDimensionNode = {}
    for framework in _FRAMEWORKS:
        points = _as_string_list(raw.get(framework, []))
        if points:
            node[framework] = points
    return node


def _as_int(raw: Any) -> int:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return 0
        try:
            return int(float(text))
        except ValueError:
            return 0
    return 0


def _as_string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []

    items: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text:
            items.append(text)
    return items


_FRAMEWORKS: tuple[FrameworkName, ...] = ("langgraph", "autogen", "crewai", "common")
