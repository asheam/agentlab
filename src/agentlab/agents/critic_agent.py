from __future__ import annotations

import json
import re
from typing import Any, Literal, cast
from urllib.parse import urlparse

from agentlab.agents.research_dimensions import CONTENT_DIMENSION_KEYWORDS, DIMENSIONS
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, ValidationError

from agentlab.core.agent import Agent, ServiceName
from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message
from agentlab.models.base import BaseModel, LLMMessage
from agentlab.workspace.research_workspace import (
    CritiquePayload,
    read_notes,
    write_critique,
)


CriticMode = Literal["auto", "rule", "llm"]


class CriticAgent(Agent):
    def __init__(self, model: BaseModel | None = None, critic_mode: CriticMode = "auto") -> None:
        super().__init__(
            name="critic",
            role="critic",
            system_prompt="Evaluate note quality and identify coverage gaps.",
            model=model,
        )
        if critic_mode not in {"auto", "rule", "llm"}:
            raise ValueError("critic_mode must be one of: auto, rule, llm")
        self.critic_mode = critic_mode

    @property
    def required_services(self) -> set[ServiceName]:
        return {"blackboard"}

    def run(self, message: Message, context: RuntimeContext) -> Message:
        if context.blackboard is None:
            raise RuntimeError("blackboard is required for CriticAgent")

        notes = read_notes(context.blackboard)
        rule_critique = _build_critique(notes)

        critique = rule_critique
        assessment_mode = "rule"
        model_fallback_reason: str | None = None

        if self._should_attempt_model():
            llm_critique, llm_error = _build_critique_with_model(notes, self.model)
            if llm_critique is not None:
                critique = _merge_rule_and_llm_critique(rule_critique, llm_critique)
                assessment_mode = "llm"
            else:
                assessment_mode = "rule_fallback"
                model_fallback_reason = llm_error
        elif self.critic_mode == "llm":
            assessment_mode = "rule_fallback"
            model_fallback_reason = "critic_mode_llm_but_model_missing"

        critique["assessment_mode"] = assessment_mode
        if model_fallback_reason:
            critique["model_fallback_reason"] = model_fallback_reason

        write_critique(
            context.blackboard,
            critique=cast(CritiquePayload, critique),
            author=self.name,
        )
        return Message(
            sender=self.name,
            receiver=message.sender,
            content="Generated critique for current notes.",
            type="response",
            metadata=critique,
        )

    def _should_attempt_model(self) -> bool:
        if self.critic_mode == "rule":
            return False
        if self.model is None:
            return False
        return self.critic_mode in {"auto", "llm"}


class _LLMScores(PydanticBaseModel):
    overall: float = Field(ge=0, le=100)
    coverage: float = Field(ge=0, le=100)
    evidence: float = Field(ge=0, le=100)
    specificity: float = Field(ge=0, le=100)
    balance: float = Field(ge=0, le=100)


class _LLMStats(PydanticBaseModel):
    key_points_count: int = Field(ge=0)
    references_count: int = Field(ge=0)
    missing_frameworks: list[str] = Field(default_factory=list)


class _LLMCritique(PydanticBaseModel):
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    verdict: Literal["acceptable", "need_improvement"]
    scores: _LLMScores
    stats: _LLMStats
    recommendations: list[str] = Field(default_factory=list)


def _build_critique(notes: Any) -> dict[str, Any]:
    summary, key_points, references = _extract_notes_fields(notes)
    corpus = " ".join(key_points + [summary]).lower()
    source_domains = _extract_source_domains(references)
    dimension_coverage_score, covered_dimensions, missing_dimensions = _score_dimension_coverage(
        notes=notes,
        key_points=key_points,
    )
    frameworks = ["langgraph", "autogen", "crewai"]
    mention_counts = {name: corpus.count(name) for name in frameworks}
    missing_frameworks = [name for name, count in mention_counts.items() if count == 0]
    comparative_points = _count_comparative_points(key_points, frameworks)
    limitation_points = _count_keyword_points(key_points, _LIMITATION_KEYWORDS)
    actionable_points = _count_keyword_points(key_points, _ACTIONABLE_KEYWORDS)

    coverage_score = _score_coverage(frameworks, missing_frameworks)
    evidence_score = _score_evidence(key_points, references, source_domains)
    specificity_score = _score_specificity(key_points)
    balance_score = _score_balance(mention_counts)
    reasoning_depth_score = _score_reasoning_depth(
        comparative_points=comparative_points,
        limitation_points=limitation_points,
        actionable_points=actionable_points,
    )
    overall_score = round(
        coverage_score * 0.30
        + evidence_score * 0.25
        + specificity_score * 0.15
        + balance_score * 0.15
        + reasoning_depth_score * 0.15,
        1,
    )

    strengths = _build_strengths(
        missing_frameworks=missing_frameworks,
        key_points=key_points,
        references=references,
        specificity_score=specificity_score,
        comparative_points=comparative_points,
        limitation_points=limitation_points,
        source_domain_count=len(source_domains),
        dimension_coverage_score=dimension_coverage_score,
    )
    gaps = _build_gaps(
        missing_frameworks=missing_frameworks,
        key_points=key_points,
        references=references,
        specificity_score=specificity_score,
        balance_score=balance_score,
        comparative_points=comparative_points,
        limitation_points=limitation_points,
        actionable_points=actionable_points,
        source_domain_count=len(source_domains),
        missing_dimensions=missing_dimensions,
    )
    recommendations = _build_recommendations(
        missing_frameworks=missing_frameworks,
        key_points=key_points,
        references=references,
        specificity_score=specificity_score,
        comparative_points=comparative_points,
        limitation_points=limitation_points,
        actionable_points=actionable_points,
        source_domain_count=len(source_domains),
        missing_dimensions=missing_dimensions,
    )
    verdict = (
        "acceptable"
        if (
            not missing_frameworks
            and comparative_points >= 1
            and overall_score >= 70.0
        )
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
            "reasoning_depth": reasoning_depth_score,
            "dimension_coverage": dimension_coverage_score,
        },
        "stats": {
            "key_points_count": len(key_points),
            "references_count": len(references),
            "missing_frameworks": missing_frameworks,
            "comparative_points_count": comparative_points,
            "limitation_points_count": limitation_points,
            "actionable_points_count": actionable_points,
            "source_domains_count": len(source_domains),
            "source_domains": source_domains,
            "dimensions_covered_count": len(covered_dimensions),
            "dimensions_covered": covered_dimensions,
            "dimensions_missing": missing_dimensions,
        },
        "recommendations": recommendations,
    }


def _build_critique_with_model(
    notes: Any,
    model: BaseModel | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if model is None:
        return None, "critic_model_missing_generate"

    prompt = (
        "你是技术评审专家。请基于给定 notes 生成 JSON 评审结果。"
        "输出必须是纯 JSON，不要输出额外文字。\n"
        "要求字段：strengths(list[str])、gaps(list[str])、"
        "verdict(acceptable|need_improvement)、"
        "scores({overall,coverage,evidence,specificity,balance} 0-100)、"
        "stats({key_points_count,references_count,missing_frameworks})、"
        "recommendations(list[str])。\n"
        "notes_json:\n"
        f"{json.dumps(notes, ensure_ascii=False, indent=2)}"
    )

    try:
        response = model.generate(
            [
                LLMMessage(
                    role="system",
                    content="You are a strict evaluator and must output JSON only.",
                ),
                LLMMessage(role="user", content=prompt),
            ]
        )
    except Exception as exc:
        return None, f"critic_model_call_failed: {exc}"

    text = response.content.strip()
    if not text:
        return None, "critic_model_empty_output"

    payload = _extract_json_payload(text)
    if payload is None:
        return None, "critic_model_invalid_json"

    try:
        parsed = _LLMCritique.model_validate(payload)
    except ValidationError as exc:
        return None, f"critic_model_schema_validation_failed: {exc}"

    return parsed.model_dump(), None


def _merge_rule_and_llm_critique(
    rule_critique: dict[str, Any],
    llm_critique: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(llm_critique)
    merged["rule_backup"] = {
        "verdict": rule_critique.get("verdict"),
        "scores": rule_critique.get("scores", {}),
        "stats": rule_critique.get("stats", {}),
    }
    return merged


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    candidates: list[str] = []

    block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if block_match:
        candidates.append(block_match.group(1))

    candidates.append(text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    return None


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


def _score_evidence(
    key_points: list[str],
    references: list[str],
    source_domains: list[str],
) -> float:
    point_component = min(len(key_points), 8) / 8 * 45
    ref_component = min(len(references), 8) / 8 * 30
    domain_component = min(len(source_domains), 4) / 4 * 25
    return round(point_component + ref_component + domain_component, 1)


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


def _score_reasoning_depth(
    comparative_points: int,
    limitation_points: int,
    actionable_points: int,
) -> float:
    comparative_component = min(comparative_points, 4) / 4 * 45
    limitation_component = min(limitation_points, 3) / 3 * 30
    actionable_component = min(actionable_points, 3) / 3 * 25
    return round(comparative_component + limitation_component + actionable_component, 1)


def _build_strengths(
    missing_frameworks: list[str],
    key_points: list[str],
    references: list[str],
    specificity_score: float,
    comparative_points: int,
    limitation_points: int,
    source_domain_count: int,
    dimension_coverage_score: float,
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
    if comparative_points >= 2:
        strengths.append("包含明确的框架横向对比结论")
    if limitation_points >= 1:
        strengths.append("已讨论部分限制条件或风险")
    if source_domain_count >= 2:
        strengths.append("引用来源具备一定多样性")
    if dimension_coverage_score >= 80:
        strengths.append("关键评估维度覆盖较完整")
    if not strengths:
        strengths.append("已形成基础研究笔记")
    return strengths


def _build_gaps(
    missing_frameworks: list[str],
    key_points: list[str],
    references: list[str],
    specificity_score: float,
    balance_score: float,
    comparative_points: int,
    limitation_points: int,
    actionable_points: int,
    source_domain_count: int,
    missing_dimensions: list[str],
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
    if comparative_points < 2:
        gaps.append("横向对比不足，需增加至少 2 条直接比较结论")
    if limitation_points < 1:
        gaps.append("缺少对限制条件与失败风险的讨论")
    if actionable_points < 1:
        gaps.append("缺少面向选型决策的可执行建议")
    if len(references) >= 2 and source_domain_count < 2:
        gaps.append("引用来源过于集中，建议增加跨来源交叉验证")
    if missing_dimensions:
        gaps.append(f"维度覆盖不足，缺少: {', '.join(missing_dimensions)}")
    if not gaps:
        gaps.append("覆盖完整度良好")
    return gaps


def _build_recommendations(
    missing_frameworks: list[str],
    key_points: list[str],
    references: list[str],
    specificity_score: float,
    comparative_points: int,
    limitation_points: int,
    actionable_points: int,
    source_domain_count: int,
    missing_dimensions: list[str],
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
    if comparative_points < 3:
        recommendations.append("补充至少 3 条“同一维度下三框架对比”的直接结论。")
    if limitation_points < 1:
        recommendations.append("为每个框架补充“失败场景/限制条件/代价”一句话。")
    if actionable_points < 1:
        recommendations.append("增加一段“团队规模与目标约束 -> 选型建议”的决策说明。")
    if len(references) >= 2 and source_domain_count < 2:
        recommendations.append("增加不同来源的证据（官方文档、社区实践、第三方评测）。")
    if missing_dimensions:
        recommendations.append(
            "按核心范式、协作方式、状态记忆、适用场景、权衡代价五个维度补齐证据。"
        )
    if not recommendations:
        recommendations.append("可进一步加入性能与维护成本的量化对比。")
    return recommendations


_COMPARATIVE_KEYWORDS = [
    "vs",
    "versus",
    "compare",
    "comparison",
    "difference",
    "trade-off",
    "tradeoff",
    "对比",
    "相比",
    "差异",
    "优缺点",
]

_LIMITATION_KEYWORDS = [
    "limitation",
    "drawback",
    "risk",
    "cost",
    "overhead",
    "complexity",
    "限制",
    "缺点",
    "风险",
    "成本",
    "复杂",
    "代价",
]

_ACTIONABLE_KEYWORDS = [
    "recommend",
    "should",
    "choose",
    "decision",
    "适用",
    "场景",
    "建议",
    "选型",
    "优先",
    "决策",
]

def _score_dimension_coverage(
    notes: Any,
    key_points: list[str],
) -> tuple[float, list[str], list[str]]:
    dimensions = list(DIMENSIONS)
    covered: set[str] = set()

    if isinstance(notes, dict):
        structured = notes.get("structured")
        if isinstance(structured, dict):
            dimension_node = structured.get("dimensions")
            if isinstance(dimension_node, dict):
                for dimension in dimensions:
                    node = dimension_node.get(dimension)
                    if not isinstance(node, dict):
                        continue
                    for framework in ("langgraph", "autogen", "crewai", "common"):
                        entries = node.get(framework, [])
                        if isinstance(entries, list) and any(str(item).strip() for item in entries):
                            covered.add(dimension)
                            break

    if not covered:
        covered = _infer_dimensions_from_points(key_points)

    covered_list = [dimension for dimension in dimensions if dimension in covered]
    missing = [dimension for dimension in dimensions if dimension not in covered]
    score = round(len(covered_list) / len(dimensions) * 100, 1) if dimensions else 0.0
    return score, covered_list, missing


def _infer_dimensions_from_points(key_points: list[str]) -> set[str]:
    covered: set[str] = set()
    for point in key_points:
        lowered = point.lower()
        for dimension, keywords in CONTENT_DIMENSION_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                covered.add(dimension)
    return covered


def _count_comparative_points(key_points: list[str], frameworks: list[str]) -> int:
    hits = 0
    for point in key_points:
        lowered = point.lower()
        mentioned = sum(1 for framework in frameworks if framework in lowered)
        has_keyword = any(keyword in lowered for keyword in _COMPARATIVE_KEYWORDS)
        if mentioned >= 2 or (mentioned >= 1 and has_keyword):
            hits += 1
    return hits


def _count_keyword_points(key_points: list[str], keywords: list[str]) -> int:
    hits = 0
    for point in key_points:
        lowered = point.lower()
        if any(keyword in lowered for keyword in keywords):
            hits += 1
    return hits


def _extract_source_domains(references: list[str]) -> list[str]:
    domains: list[str] = []
    for reference in references:
        value = reference.strip()
        if not value:
            continue
        parsed = urlparse(value)
        host = parsed.netloc.lower().strip()
        if host.startswith("www."):
            host = host[4:]
        if host:
            domains.append(host)
            continue
        if "/" in value and " " not in value:
            domains.append(value.split("/", 1)[0].lower())
    return list(dict.fromkeys(domains))
