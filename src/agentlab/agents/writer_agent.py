from __future__ import annotations

from typing import Any

from agentlab.core.agent import Agent
from agentlab.core.context import RuntimeContext
from agentlab.core.event import Event
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
        if self.model is not None:
            generated, model_error = _build_report_with_model(
                topic=topic,
                plan=plan,
                notes=notes,
                critique=critique,
                search_results=search_results,
                model=self.model,
            )
            if generated:
                report = generated
                _record_model_event(
                    context=context,
                    success=True,
                    error=None,
                    input_text=topic,
                    output_text="model report generated",
                )
            else:
                fallback_reason = model_error or "Writer model returned empty output."
                report = _append_fallback_note(report, fallback_reason)
                _record_model_event(
                    context=context,
                    success=False,
                    error=fallback_reason,
                    input_text=topic,
                    output_text="fallback to template report",
                )

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

    lines.append("## Search Mode Summary")
    summary = _summarize_search_modes(search_results)
    lines.append(f"- Total queries: {summary['total_queries']}")
    lines.append(f"- Real results: {summary['real_count']}")
    lines.append(f"- Mock results: {summary['mock_count']}")
    lines.append(f"- Fallback used: {summary['fallback_count']}")
    lines.append(f"- DuckDuckGo hits: {summary['duckduckgo_hits']}")
    lines.append(f"- Wikipedia hits: {summary['wikipedia_hits']}")
    if summary["fallback_reasons"]:
        lines.append(f"- Fallback reasons: {', '.join(summary['fallback_reasons'])}")
    if summary["error_count"] > 0:
        lines.append(f"- Tool errors: {summary['error_count']}")
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


def _build_report_with_model(
    topic: str,
    plan: Any,
    notes: Any,
    critique: Any,
    search_results: Any,
    model: Any,
) -> tuple[str | None, str | None]:
    if not hasattr(model, "generate"):
        return None, "Model does not implement generate()."

    prompt = (
        "请输出 Markdown 研究报告，必须包含以下标题：\n"
        "1) # Deep Research Report\n"
        "2) ## Topic\n"
        "3) ## Research Questions\n"
        "4) ## Key Findings\n"
        "5) ## Framework Snapshot\n"
        "6) ## Critique\n"
        "7) ## References\n"
        "语言：中文，结论明确，避免冗长。\n\n"
        f"Topic: {topic}\n"
        f"Plan: {_safe_json(plan)}\n"
        f"Notes: {_safe_json(notes)}\n"
        f"Critique: {_safe_json(critique)}\n"
        f"SearchResults: {_safe_json(search_results)}\n"
    )
    try:
        output = model.generate(
            [
                {"role": "system", "content": "You are a senior technical research writer."},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as exc:
        return None, f"Writer model call failed: {exc}"

    text = output.strip()
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
        "## Critique",
        "## References",
    ]
    return [header for header in required_headers if header not in text]


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


def _summarize_search_modes(search_results: Any) -> dict[str, Any]:
    total_queries = 0
    real_count = 0
    mock_count = 0
    fallback_count = 0
    error_count = 0
    duckduckgo_hits = 0
    wikipedia_hits = 0
    fallback_reasons: list[str] = []

    if not isinstance(search_results, list):
        return {
            "total_queries": 0,
            "real_count": 0,
            "mock_count": 0,
            "fallback_count": 0,
            "error_count": 0,
            "duckduckgo_hits": 0,
            "wikipedia_hits": 0,
            "fallback_reasons": [],
        }

    for item in search_results:
        if not isinstance(item, dict):
            continue
        total_queries += 1

        if "error" in item:
            error_count += 1
            continue

        result = item.get("result")
        if not isinstance(result, dict):
            continue

        mode = result.get("mode")
        if mode == "real":
            real_count += 1
        elif mode == "mock":
            mock_count += 1

        source_hits = result.get("source_hits")
        if isinstance(source_hits, dict):
            duckduckgo_hits += int(source_hits.get("duckduckgo", 0) or 0)
            wikipedia_hits += int(source_hits.get("wikipedia", 0) or 0)

        if result.get("fallback_used") is True:
            fallback_count += 1
            reason = result.get("fallback_reason")
            if isinstance(reason, str) and reason:
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
        "fallback_reasons": unique_reasons,
    }
