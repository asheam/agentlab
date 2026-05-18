from __future__ import annotations

import argparse

from agentlab.agents.critic_agent import CriticStrategyInput, DefaultCriticStrategy
from agentlab.agents.reader_agent import DefaultReaderStrategy, ReaderStrategyInput
from agentlab.agents.search_agent import (
    DefaultSearchStrategy,
    SearchStrategyInput,
    SearchStrategyOutput,
)
from agentlab.agents.writer_agent import WriterStrategyInput
from agentlab.models.base import BaseModel
from agentlab.multi_agent.supervisor import SupervisorConfig, build_default_supervisor
from agentlab.workspace.research_workspace import NotesPayload


class FocusedPlannerStrategy:
    """A planner strategy that enforces a fixed 5-question research frame."""

    def build_plan(self, topic: str, model: BaseModel | None) -> list[str]:
        del model
        return [
            f"{topic} 的核心设计理念是什么？",
            f"{topic} 的任务编排机制如何工作？",
            f"{topic} 在状态管理与可观测性上的关键差异是什么？",
            f"{topic} 的工程落地成本和扩展性如何？",
            f"{topic} 在中小团队中的选型建议是什么？",
        ]


class TargetedSearchStrategy:
    """Use default tool-calling behavior but limit to top-N plan items."""

    def __init__(self, max_questions: int = 4) -> None:
        self.max_questions = max_questions
        self._delegate = DefaultSearchStrategy()

    def collect(self, request: SearchStrategyInput) -> SearchStrategyOutput:
        narrowed = SearchStrategyInput(
            topic=request.topic,
            plan=request.plan[: self.max_questions],
            tool_name=request.tool_name,
            context=request.context,
            agent_name=request.agent_name,
        )
        return self._delegate.collect(narrowed)


class ConciseReaderStrategy:
    """Post-process default notes into a compact structure."""

    def __init__(self) -> None:
        self._delegate = DefaultReaderStrategy()

    def build_notes(self, request: ReaderStrategyInput) -> NotesPayload:
        notes = self._delegate.build_notes(request)
        key_points = list(notes.get("key_points", []))[:6]
        references = list(notes.get("references", []))[:8]
        summary = str(notes.get("summary", ""))
        structured = notes.get("structured", {})
        return NotesPayload(
            key_points=key_points,
            references=references,
            summary=summary,
            structured=structured,
        )


class RuleOnlyCriticStrategy:
    """Force deterministic rule-based critique regardless of model presence."""

    def __init__(self) -> None:
        self._delegate = DefaultCriticStrategy()

    def build_critique(self, request: CriticStrategyInput) -> dict[str, object]:
        forced_rule = CriticStrategyInput(
            notes=request.notes,
            model=request.model,
            critic_mode="rule",
        )
        return self._delegate.build_critique(forced_rule)


class ExecutiveWriterStrategy:
    """A compact writer strategy for executive-style markdown summaries."""

    def build_report(self, request: WriterStrategyInput) -> str:
        key_points = request.notes.get("key_points", [])
        references = request.notes.get("references", [])
        verdict = request.critique.get("verdict", "unknown")

        lines: list[str] = []
        lines.append("# Deep Research Report")
        lines.append("")
        lines.append("## Topic")
        lines.append(request.topic)
        lines.append("")
        lines.append("## Executive Verdict")
        lines.append(f"- Verdict: {verdict}")
        lines.append(f"- Questions processed: {len(request.plan)}")
        lines.append(f"- Key findings captured: {len(key_points)}")
        lines.append("")
        lines.append("## Research Questions")
        for idx, question in enumerate(request.plan, start=1):
            lines.append(f"{idx}. {question}")
        lines.append("")
        lines.append("## Key Findings")
        if key_points:
            for point in key_points[:8]:
                lines.append(f"- {point}")
        else:
            lines.append("- No key findings captured.")
        lines.append("")
        lines.append("## References")
        if references:
            for ref in references[:10]:
                lines.append(f"- {ref}")
        else:
            lines.append("- No references.")

        return "\n".join(lines).strip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deep research with custom strategies across planner/search/reader/critic/writer."
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default="研究 LangGraph、AutoGen、CrewAI 的区别",
        help="Research topic",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/custom_strategies",
        help="Output directory for generated artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config = SupervisorConfig(
        output_dir=args.output_dir,
        search_mode="mock",
        critic_mode="rule",
        planner_strategy=FocusedPlannerStrategy(),
        search_strategy=TargetedSearchStrategy(max_questions=4),
        reader_strategy=ConciseReaderStrategy(),
        critic_strategy=RuleOnlyCriticStrategy(),
        writer_strategy=ExecutiveWriterStrategy(),
    )
    supervisor = build_default_supervisor(config=config)
    outputs = supervisor.run(args.topic)

    print("Custom strategy demo completed")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
