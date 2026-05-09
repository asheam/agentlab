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
    summary, key_points, references = _extract_notes_fields(notes)
    corpus = " ".join(key_points + [summary]).lower()
    frameworks = ["langgraph", "autogen", "crewai"]
    mention_counts = {name: corpus.count(name) for name in frameworks}
    missing_frameworks = [name for name, count in mention_counts.items() if count == 0]

    coverage_score = _score_coverage(frameworks, missing_frameworks)
    evidence_score = _score_evidence(key_points, references)
    specificity_score = _score_specificity(key_points)
    balance_score = _score_balance(mention_counts)
    overall_score = round(
        coverage_score * 0.40
        + evidence_score * 0.25
        + specificity_score * 0.20
        + balance_score * 0.15,
        1,
    )

    strengths = _build_strengths(
        missing_frameworks=missing_frameworks,
        key_points=key_points,
        references=references,
        specificity_score=specificity_score,
    )
    gaps = _build_gaps(
        missing_frameworks=missing_frameworks,
        key_points=key_points,
        references=references,
        specificity_score=specificity_score,
        balance_score=balance_score,
    )
    recommendations = _build_recommendations(
        missing_frameworks=missing_frameworks,
        key_points=key_points,
        references=references,
        specificity_score=specificity_score,
    )
    verdict = (
        "acceptable"
        if (not missing_frameworks and overall_score >= 70.0)
        else "need_improvement"
    )

    return {
        "strengths": strengths,
        "gaps": gaps,
        "verdict": verdict,
        "scores": {
            "overall": overall_score,
            "coverage": coverage_score,
            "evidence": evidence_score,
            "specificity": specificity_score,
            "balance": balance_score,
        },
        "stats": {
            "key_points_count": len(key_points),
            "references_count": len(references),
            "missing_frameworks": missing_frameworks,
        },
        "recommendations": recommendations,
    }


def _extract_notes_fields(notes: Any) -> tuple[str, list[str], list[str]]:
    summary = ""
    key_points: list[str] = []
    references: list[str] = []
    if not isinstance(notes, dict):
        return summary, key_points, references

    summary = str(notes.get("summary", ""))

    raw_points = notes.get("key_points", [])
    if isinstance(raw_points, list):
        key_points = [str(item).strip() for item in raw_points if str(item).strip()]

    raw_references = notes.get("references", [])
    if isinstance(raw_references, list):
        references = [str(item).strip() for item in raw_references if str(item).strip()]

    return summary, key_points, references


def _score_coverage(frameworks: list[str], missing_frameworks: list[str]) -> float:
    if not frameworks:
        return 0.0
    covered = len(frameworks) - len(missing_frameworks)
    return round(covered / len(frameworks) * 100, 1)


def _score_evidence(key_points: list[str], references: list[str]) -> float:
    point_component = min(len(key_points), 8) / 8 * 60
    ref_component = min(len(references), 8) / 8 * 40
    return round(point_component + ref_component, 1)


def _score_specificity(key_points: list[str]) -> float:
    if not key_points:
        return 0.0

    informative_points = sum(1 for point in key_points if len(point) >= 60)
    keywords = [
        "difference",
        "compare",
        "workflow",
        "state",
        "tool",
        "observability",
        "memory",
        "优缺点",
        "适用",
        "场景",
    ]
    keyword_hits = 0
    for point in key_points:
        lowered = point.lower()
        if any(keyword in lowered for keyword in keywords):
            keyword_hits += 1

    length_component = min(informative_points, 6) / 6 * 70
    keyword_component = min(keyword_hits, 6) / 6 * 30
    return round(length_component + keyword_component, 1)


def _score_balance(mention_counts: dict[str, int]) -> float:
    counts = list(mention_counts.values())
    if not counts or all(value == 0 for value in counts):
        return 0.0
    if any(value == 0 for value in counts):
        covered = sum(1 for value in counts if value > 0)
        return round(covered / len(counts) * 60, 1)

    max_count = max(counts)
    min_count = min(counts)
    if max_count == 0:
        return 0.0
    return round(min_count / max_count * 100, 1)


def _build_strengths(
    missing_frameworks: list[str],
    key_points: list[str],
    references: list[str],
    specificity_score: float,
) -> list[str]:
    strengths: list[str] = []
    if not missing_frameworks:
        strengths.append("已覆盖 LangGraph、AutoGen、CrewAI 三个框架")
    if len(key_points) >= 5:
        strengths.append("关键要点数量充足")
    if len(references) >= 3:
        strengths.append("参考来源数量充足")
    if specificity_score >= 60:
        strengths.append("要点具备较好的具体性与可操作性")
    if not strengths:
        strengths.append("已形成基础研究笔记")
    return strengths


def _build_gaps(
    missing_frameworks: list[str],
    key_points: list[str],
    references: list[str],
    specificity_score: float,
    balance_score: float,
) -> list[str]:
    gaps: list[str] = []
    gaps.extend(f"缺少对 {name} 的充分覆盖" for name in missing_frameworks)
    if len(key_points) < 4:
        gaps.append("关键要点偏少，建议补充至少 4 条可验证结论")
    if len(references) < 2:
        gaps.append("参考来源偏少，建议补充多来源证据")
    if specificity_score < 50:
        gaps.append("要点较笼统，建议补充具体比较维度与结论")
    if balance_score < 50:
        gaps.append("三框架讨论不均衡，建议补齐横向对比")
    if not gaps:
        gaps.append("覆盖完整度良好")
    return gaps


def _build_recommendations(
    missing_frameworks: list[str],
    key_points: list[str],
    references: list[str],
    specificity_score: float,
) -> list[str]:
    recommendations: list[str] = []
    if missing_frameworks:
        targets = ", ".join(missing_frameworks)
        recommendations.append(f"优先补充 {targets} 的设计理念、编排方式与适用场景。")
    if len(references) < 3:
        recommendations.append("增加官方文档或技术博客来源，提升结论可追溯性。")
    if len(key_points) < 5:
        recommendations.append("按“架构、工具、状态、可观测性、落地成本”补齐对比维度。")
    if specificity_score < 60:
        recommendations.append("每条结论补充一句“适用场景+限制条件”。")
    if not recommendations:
        recommendations.append("可进一步加入性能与维护成本的量化对比。")
    return recommendations
