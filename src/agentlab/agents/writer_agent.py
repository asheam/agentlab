from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from agentlab.core.agent import Agent, ServiceName
from agentlab.core.context import RuntimeContext
from agentlab.core.event import Event
from agentlab.core.message import Message
from agentlab.models.base import BaseModel, LLMMessage
from agentlab.workspace.research_workspace import (
    CritiquePayload,
    NotesPayload,
    SearchResultItem,
    critique_assessment_mode,
    critique_gaps,
    critique_recommendations,
    critique_scores,
    critique_stats,
    critique_strengths,
    critique_verdict,
    notes_dimension_points,
    notes_framework_points,
    notes_key_points,
    notes_references,
    notes_summary,
    read_critique,
    read_notes,
    read_plan,
    read_search_results,
    search_result_error,
    search_result_fallback_reason,
    search_result_fallback_used,
    search_result_issues,
    search_result_mode,
    search_result_payload,
    search_result_provider_errors,
    search_result_source_hits,
    write_report,
)


@dataclass(frozen=True)
class WriterStrategyInput:
    topic: str
    plan: list[str]
    search_results: list[SearchResultItem]
    notes: NotesPayload
    critique: CritiquePayload
    model: BaseModel | None
    context: RuntimeContext


class WriterStrategy(Protocol):
    def build_report(self, request: WriterStrategyInput) -> str:
        """Compose final markdown report."""


class DefaultWriterStrategy:
    def build_report(self, request: WriterStrategyInput) -> str:
        report = _build_report(
            request.topic,
            request.plan,
            request.search_results,
            request.notes,
            request.critique,
        )
        if request.model is None:
            return report

        generated, model_error = _build_report_with_model(
            topic=request.topic,
            plan=request.plan,
            notes=request.notes,
            critique=request.critique,
            search_results=request.search_results,
            model=request.model,
        )
        if generated:
            _record_model_event(
                context=request.context,
                success=True,
                error=None,
                input_text=request.topic,
                output_text="model report generated",
            )
            return generated

        fallback_reason = model_error or "Writer model returned empty output."
        _record_model_event(
            context=request.context,
            success=False,
            error=fallback_reason,
            input_text=request.topic,
            output_text="fallback to template report",
        )
        return _append_fallback_note(report, fallback_reason)


class WriterAgent(Agent):
    def __init__(
        self,
        model: BaseModel | None = None,
        strategy: WriterStrategy | None = None,
    ) -> None:
        super().__init__(
            name="writer",
            role="writer",
            system_prompt="Write final markdown report from shared workspace.",
            model=model,
        )
        self.strategy = strategy or DefaultWriterStrategy()

    @property
    def required_services(self) -> set[ServiceName]:
        return {"blackboard"}

    def run(self, message: Message, context: RuntimeContext) -> Message:
        if context.blackboard is None:
            raise RuntimeError("blackboard is required for WriterAgent")

        topic = message.metadata.get("topic") if isinstance(message.metadata, dict) else None
        if not isinstance(topic, str) or not topic.strip():
            topic = message.content.strip() or "Untitled Topic"

        plan = read_plan(context.blackboard)
        search_results = read_search_results(context.blackboard)
        notes = read_notes(context.blackboard)
        critique = read_critique(context.blackboard)

        report = self.strategy.build_report(
            WriterStrategyInput(
                topic=topic,
                plan=plan,
                search_results=search_results,
                notes=notes,
                critique=critique,
                model=self.model,
                context=context,
            )
        )

        write_report(context.blackboard, report=report, author=self.name)

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
    plan: list[str],
    search_results: list[SearchResultItem],
    notes: NotesPayload,
    critique: CritiquePayload,
) -> str:
    key_points = notes_key_points(notes)
    references = notes_references(notes)
    summary_text = notes_summary(notes)
    verdict = critique_verdict(critique)
    summary = _summarize_search_modes(search_results)

    lines: list[str] = []
    lines.append("# Deep Research Report")
    lines.append("")
    lines.append("> Generated by AgentLab v0.1")
    lines.append("")
    lines.append(f"## Topic\n{topic}")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append(f"- Verdict: {verdict}")
    lines.append(
        "- Retrieval mix: "
        f"real={summary['real_count']}, mock={summary['mock_count']}, fallback={summary['fallback_count']}"
    )
    summary_line = _truncate_text(str(summary_text), max_chars=220)
    lines.append(f"- Research summary: {summary_line or '暂无可用摘要。'}")
    lines.append("")

    lines.append("## Research Questions")
    if plan:
        for idx, item in enumerate(plan, start=1):
            lines.append(f"{idx}. {item}")
    else:
        lines.append("1. 无可用研究问题")
    lines.append("")

    lines.append("## Framework Snapshot")
    lines.append("| Framework | Observation |")
    lines.append("| --- | --- |")
    for framework, evidence in _framework_snapshot(notes, key_points):
        lines.append(f"| {framework} | {_escape_table_cell(evidence)} |")
    lines.append("")

    lines.append("## Comparison Matrix")
    lines.append("| Dimension | LangGraph | AutoGen | CrewAI |")
    lines.append("| --- | --- | --- | --- |")
    for dimension, langgraph, autogen, crewai in _comparison_matrix_rows(notes, key_points):
        lines.append(
            f"| {dimension} | {_escape_table_cell(langgraph)} | "
            f"{_escape_table_cell(autogen)} | {_escape_table_cell(crewai)} |"
        )
    lines.append(
        "_Italic cells indicate fallback defaults when direct evidence is unavailable._"
    )
    lines.append("")

    lines.append("## Key Findings")
    if key_points:
        for point in key_points[:10]:
            lines.append(f"- {_truncate_text(str(point), max_chars=260)}")
    else:
        lines.append("- 暂无检索要点")
    lines.append("")

    lines.append("## Search Mode Summary")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Total queries | {summary['total_queries']} |")
    lines.append(f"| Real results | {summary['real_count']} |")
    lines.append(f"| Mock results | {summary['mock_count']} |")
    lines.append(f"| Fallback used | {summary['fallback_count']} |")
    lines.append(f"| DuckDuckGo hits | {summary['duckduckgo_hits']} |")
    lines.append(f"| Wikipedia hits | {summary['wikipedia_hits']} |")
    lines.append(f"| Tavily hits | {summary['tavily_hits']} |")
    lines.append(f"| DuckDuckGo errors | {summary['duckduckgo_errors']} |")
    lines.append(f"| Wikipedia errors | {summary['wikipedia_errors']} |")
    lines.append(f"| Tavily errors | {summary['tavily_errors']} |")
    if summary["fallback_reasons"]:
        lines.append("")
        for idx, reason in enumerate(summary["fallback_reasons"][:5], start=1):
            lines.append(f"- Fallback reason {idx}: {_truncate_text(reason, max_chars=180)}")
    if summary["error_count"] > 0:
        lines.append(f"- Tool errors: {summary['error_count']}")
    lines.append("")

    lines.append("## Critique")
    mode = critique_assessment_mode(critique)
    if mode is not None:
        lines.append(f"- Assessment mode: {mode}")
    for strength in critique_strengths(critique):
        lines.append(f"- Strength: {strength}")
    for gap in critique_gaps(critique):
        lines.append(f"- Gap: {gap}")
    scores = critique_scores(critique)
    overall = scores.get("overall", "n/a")
    coverage = scores.get("coverage", "n/a")
    evidence = scores.get("evidence", "n/a")
    specificity = scores.get("specificity", "n/a")
    balance = scores.get("balance", "n/a")
    reasoning_depth = scores.get("reasoning_depth", "n/a")
    dimension_coverage = scores.get("dimension_coverage", "n/a")
    lines.append(
        "- Scores: "
        f"overall={overall}, coverage={coverage}, evidence={evidence}, "
        f"specificity={specificity}, balance={balance}, "
        f"reasoning_depth={reasoning_depth}, dimension_coverage={dimension_coverage}"
    )
    stats = critique_stats(critique)
    comparative = stats.get("comparative_points_count", "n/a")
    limitation = stats.get("limitation_points_count", "n/a")
    actionable = stats.get("actionable_points_count", "n/a")
    domains = stats.get("source_domains_count", "n/a")
    dimensions = stats.get("dimensions_covered_count", "n/a")
    lines.append(
        "- Signals: "
        f"comparative_points={comparative}, limitation_points={limitation}, "
        f"actionable_points={actionable}, source_domains={domains}, "
        f"dimensions_covered={dimensions}"
    )
    for rec in critique_recommendations(critique)[:4]:
        lines.append(f"- Recommendation: {rec}")
    lines.append(f"- Verdict: {verdict}")
    lines.append("")

    lines.append("## References")
    if references:
        for ref in references[:12]:
            lines.append(f"- {_format_reference(ref)}")
    else:
        lines.append("- 无引用")

    return "\n".join(lines).strip() + "\n"


def _safe_json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)


def _truncate_text(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def _escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _format_reference(reference: Any) -> str:
    text = str(reference).strip()
    if text.startswith(("http://", "https://")):
        return f"[{text}]({text})"
    return text


def _framework_snapshot(notes: NotesPayload, key_points: list[str]) -> list[tuple[str, str]]:
    framework_points = _framework_points_for_matrix(notes, key_points)
    snapshots: list[tuple[str, str]] = []
    for framework, label in (
        ("langgraph", "LangGraph"),
        ("autogen", "AutoGen"),
        ("crewai", "CrewAI"),
    ):
        points = framework_points.get(framework, [])
        if not points:
            continue
        observation = _representative_framework_observation(framework, points)
        snapshots.append((label, observation))
    if not snapshots:
        snapshots.append(("General", "No framework-specific findings."))
    return snapshots


def _build_report_with_model(
    topic: str,
    plan: Any,
    notes: Any,
    critique: Any,
    search_results: Any,
    model: BaseModel,
) -> tuple[str | None, str | None]:
    prompt = (
        "请输出 Markdown 研究报告，必须包含以下标题：\n"
        "1) # Deep Research Report\n"
        "2) ## Topic\n"
        "3) ## Research Questions\n"
        "4) ## Key Findings\n"
        "5) ## Framework Snapshot\n"
        "6) ## Comparison Matrix\n"
        "7) ## Critique\n"
        "8) ## References\n"
        "语言：中文，结论明确，避免冗长。\n\n"
        f"Topic: {topic}\n"
        f"Plan: {_safe_json(plan)}\n"
        f"Notes: {_safe_json(notes)}\n"
        f"Critique: {_safe_json(critique)}\n"
        f"SearchResults: {_safe_json(search_results)}\n"
    )
    try:
        response = model.generate(
            [
                LLMMessage(role="system", content="You are a senior technical research writer."),
                LLMMessage(role="user", content=prompt),
            ]
        )
    except Exception as exc:
        return None, f"Writer model call failed: {exc}"

    text = response.content.strip()
    if not text:
        return None, "Writer model returned empty output."

    missing_headers = _missing_required_headers(text)
    if missing_headers:
        return None, f"Writer model output missing required headers: {', '.join(missing_headers)}"

    if not text.endswith("\n"):
        text += "\n"
    return text, None


def _missing_required_headers(text: str) -> list[str]:
    required_headers = [
        "# Deep Research Report",
        "## Topic",
        "## Research Questions",
        "## Key Findings",
        "## Framework Snapshot",
        "## Comparison Matrix",
        "## Critique",
        "## References",
    ]
    return [header for header in required_headers if header not in text]


def _comparison_matrix_rows(notes: NotesPayload, key_points: list[str]) -> list[tuple[str, str, str, str]]:
    framework_points = _framework_points_for_matrix(notes, key_points)
    dimension_points = _dimension_points_for_matrix(notes)
    row_specs = [
        (
            "core_paradigm",
            "Core Paradigm",
            ["graph", "state", "workflow", "图", "状态机", "编排"],
            ["conversation", "dialog", "message", "对话", "消息"],
            ["role", "task", "crew", "角色", "任务"],
            {
                "langgraph": "Graph/state-machine orchestration with explicit transitions.",
                "autogen": "Conversation-loop based multi-agent collaboration.",
                "crewai": "Role/task-oriented crew execution model.",
            },
        ),
        (
            "coordination_style",
            "Coordination Style",
            ["deterministic", "node", "edge", "branch", "确定性", "节点"],
            ["loop", "turn", "chat", "交互", "轮次"],
            ["delegate", "assignment", "owner", "分工", "委派"],
            {
                "langgraph": "Deterministic node-edge workflow and branch control.",
                "autogen": "Dynamic dialogue with agent turn-taking.",
                "crewai": "Role delegation and task-centric collaboration.",
            },
        ),
        (
            "state_memory",
            "State & Memory",
            ["durable", "state", "checkpoint", "持久", "状态", "检查点"],
            ["history", "message", "memory", "历史", "记忆"],
            ["context", "task context", "上下文", "共享"],
            {
                "langgraph": "Explicit state persistence and checkpoint recovery.",
                "autogen": "Conversation history centric; memory is often external.",
                "crewai": "Lightweight shared context around tasks and roles.",
            },
        ),
        (
            "best_fit",
            "Best Fit",
            ["complex", "audit", "reliable", "复杂", "可追踪"],
            ["prototype", "experiment", "rapid", "原型", "实验"],
            ["business", "workflow", "operation", "业务", "流程"],
            {
                "langgraph": "Complex, auditable and long-running workflows.",
                "autogen": "Rapid prototyping of conversational agent systems.",
                "crewai": "Business process automation with clear ownership.",
            },
        ),
        (
            "trade_off",
            "Main Trade-off",
            ["overhead", "complexity", "cost", "复杂", "成本"],
            ["drift", "stability", "control", "漂移", "控制"],
            ["depth", "customization", "abstraction", "深度", "定制"],
            {
                "langgraph": "Higher modeling complexity and maintenance cost.",
                "autogen": "Conversation flow can drift without strict constraints.",
                "crewai": "Abstraction is simple, but deep customization can be limited.",
            },
        ),
    ]

    rows: list[tuple[str, str, str, str]] = []
    for dimension_key, dimension, lg_keywords, ag_keywords, ca_keywords, defaults in row_specs:
        rows.append(
            (
                dimension,
                _select_dimension_row_cell(
                    framework="langgraph",
                    framework_points=framework_points["langgraph"],
                    dimension_points=dimension_points[dimension_key]["langgraph"],
                    keywords=lg_keywords,
                    fallback=defaults["langgraph"],
                ),
                _select_dimension_row_cell(
                    framework="autogen",
                    framework_points=framework_points["autogen"],
                    dimension_points=dimension_points[dimension_key]["autogen"],
                    keywords=ag_keywords,
                    fallback=defaults["autogen"],
                ),
                _select_dimension_row_cell(
                    framework="crewai",
                    framework_points=framework_points["crewai"],
                    dimension_points=dimension_points[dimension_key]["crewai"],
                    keywords=ca_keywords,
                    fallback=defaults["crewai"],
                ),
            )
        )
    return rows


def _group_framework_points(key_points: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {"langgraph": [], "autogen": [], "crewai": []}

    for item in key_points:
        text = str(item).strip()
        if not text:
            continue
        lowered = text.lower()
        for name in grouped:
            if name in lowered:
                grouped[name].append(text)
    return grouped


def _framework_points_for_matrix(notes: NotesPayload, key_points: list[str]) -> dict[str, list[str]]:
    grouped = _group_framework_points(key_points)

    for framework in ("langgraph", "autogen", "crewai"):
        points = notes_framework_points(notes, framework)
        if points:
            grouped[framework] = list(dict.fromkeys(points))
    return grouped


def _select_dimension_row_cell(
    framework: str,
    framework_points: list[str],
    dimension_points: list[str],
    keywords: list[str],
    fallback: str,
) -> str:
    if dimension_points:
        return _select_framework_cell(
            framework=framework,
            points=dimension_points,
            keywords=keywords,
            fallback=fallback,
            fallback_when_no_match=False,
        )

    return _select_framework_cell(
        framework=framework,
        points=framework_points,
        keywords=keywords,
        fallback=fallback,
        fallback_when_no_match=True,
    )


def _dimension_points_for_matrix(notes: NotesPayload) -> dict[str, dict[str, list[str]]]:
    matrix_points: dict[str, dict[str, list[str]]] = {
        "core_paradigm": {"langgraph": [], "autogen": [], "crewai": []},
        "coordination_style": {"langgraph": [], "autogen": [], "crewai": []},
        "state_memory": {"langgraph": [], "autogen": [], "crewai": []},
        "best_fit": {"langgraph": [], "autogen": [], "crewai": []},
        "trade_off": {"langgraph": [], "autogen": [], "crewai": []},
    }
    for dimension_key in matrix_points:
        for framework in ("langgraph", "autogen", "crewai"):
            points = notes_dimension_points(notes, dimension_key, framework)
            if points:
                matrix_points[dimension_key][framework] = list(dict.fromkeys(points))
    return matrix_points


def _select_framework_cell(
    framework: str,
    points: list[str],
    keywords: list[str],
    fallback: str,
    fallback_when_no_match: bool = True,
) -> str:
    if not points:
        return _format_fallback_cell(fallback)

    best_candidate: str | None = None
    best_score = 0
    for point in points:
        candidate = _extract_framework_clause(point, framework) or point
        lowered = candidate.lower()
        if "strongest for" in lowered:
            continue
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > best_score:
            best_score = score
            best_candidate = candidate
            continue
        if score == best_score and score > 0 and best_candidate is not None:
            if len(candidate) > len(best_candidate):
                best_candidate = candidate

    if best_candidate is not None and best_score > 0:
        return _truncate_text(best_candidate, max_chars=88)
    if not fallback_when_no_match:
        for point in points:
            candidate = _extract_framework_clause(point, framework) or point
            if "strongest for" in candidate.lower():
                continue
            return _truncate_text(candidate, max_chars=88)
    return _format_fallback_cell(fallback)


def _extract_framework_clause(text: str, framework: str) -> str | None:
    segments = re.split(r"[;；]\s*", text)
    token = framework.lower()
    for segment in segments:
        lowered = segment.lower()
        if token not in lowered:
            continue
        cleaned = re.sub(
            rf"^\s*{re.escape(framework)}(?:\s*[:：\-]\s*|[\s,，]+)",
            "",
            segment,
            flags=re.I,
        ).strip()
        return cleaned or segment.strip()
    return None


def _representative_framework_observation(framework: str, points: list[str]) -> str:
    if not points:
        return "No framework-specific findings."
    candidate = _extract_framework_clause(points[0], framework) or points[0]
    return _truncate_text(candidate, max_chars=120)


def _format_fallback_cell(text: str) -> str:
    compact = text.strip()
    if compact.startswith("_") and compact.endswith("_") and len(compact) >= 2:
        return compact
    return f"_{compact}_"


def _append_fallback_note(report: str, reason: str) -> str:
    note = (
        "\n## Model Fallback\n"
        f"- Writer model output was not used: {reason}\n"
    )
    if report.endswith("\n"):
        return report + note
    return report + "\n" + note


def _record_model_event(
    context: RuntimeContext,
    success: bool,
    error: str | None,
    input_text: str,
    output_text: str,
) -> None:
    if context.trace_recorder is None:
        return
    context.trace_recorder.record(
        Event(
            event_type="model_call",
            agent="writer",
            input=input_text,
            output=output_text,
            success=success,
            error=error,
        )
    )


def _summarize_search_modes(search_results: list[SearchResultItem]) -> dict[str, Any]:
    total_queries = 0
    real_count = 0
    mock_count = 0
    fallback_count = 0
    error_count = 0
    duckduckgo_hits = 0
    wikipedia_hits = 0
    tavily_hits = 0
    provider_error_counts = {
        "duckduckgo": 0,
        "wikipedia": 0,
        "tavily": 0,
    }
    fallback_reasons: list[str] = []

    for item in search_results:
        total_queries += 1

        item_error = search_result_error(item)
        if item_error is not None:
            error_count += 1
            _accumulate_provider_error_counts([item_error], provider_error_counts)
            continue

        result = search_result_payload(item)
        if result is None:
            continue

        mode = search_result_mode(result)
        if mode == "real":
            real_count += 1
        elif mode == "mock":
            mock_count += 1

        source_hits = search_result_source_hits(result)
        duckduckgo_hits += source_hits.get("duckduckgo", 0)
        wikipedia_hits += source_hits.get("wikipedia", 0)
        tavily_hits += source_hits.get("tavily", 0)
        provider_errors = search_result_provider_errors(result)

        issue_texts = search_result_issues(result)

        if not issue_texts:
            fallback_reason = search_result_fallback_reason(result)
            if fallback_reason:
                issue_texts.append(fallback_reason)

        if any(provider_errors.values()):
            provider_error_counts["duckduckgo"] += provider_errors.get("duckduckgo", 0)
            provider_error_counts["wikipedia"] += provider_errors.get("wikipedia", 0)
            provider_error_counts["tavily"] += provider_errors.get("tavily", 0)
        else:
            _accumulate_provider_error_counts(issue_texts, provider_error_counts)

        if search_result_fallback_used(result):
            fallback_count += 1
            reason = search_result_fallback_reason(result)
            if reason:
                fallback_reasons.append(reason)

    unique_reasons = list(dict.fromkeys(fallback_reasons))
    return {
        "total_queries": total_queries,
        "real_count": real_count,
        "mock_count": mock_count,
        "fallback_count": fallback_count,
        "error_count": error_count,
        "duckduckgo_hits": duckduckgo_hits,
        "wikipedia_hits": wikipedia_hits,
        "tavily_hits": tavily_hits,
        "duckduckgo_errors": provider_error_counts["duckduckgo"],
        "wikipedia_errors": provider_error_counts["wikipedia"],
        "tavily_errors": provider_error_counts["tavily"],
        "fallback_reasons": unique_reasons,
    }


_PROVIDER_ERROR_PATTERN = re.compile(r"\b([a-z0-9_-]+)_error:")


def _accumulate_provider_error_counts(
    issue_texts: list[str],
    provider_error_counts: dict[str, int],
) -> None:
    for issue_text in issue_texts:
        normalized = issue_text.lower()
        for match in _PROVIDER_ERROR_PATTERN.finditer(normalized):
            provider = match.group(1)
            if provider in provider_error_counts:
                provider_error_counts[provider] += 1
