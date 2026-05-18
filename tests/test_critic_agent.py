from __future__ import annotations

from typing import Any

from agentlab.agents.critic_agent import CriticAgent, CriticStrategyInput, _build_critique
from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message
from agentlab.models.base import LLMMessage, ModelResponse
from agentlab.workspace.blackboard import Blackboard


def test_critic_flags_missing_frameworks() -> None:
    critique = _build_critique(
        {
            "summary": "Focused mostly on LangGraph workflow design.",
            "key_points": [
                "LangGraph has explicit graph transitions and deterministic control flow.",
            ],
            "references": ["https://example.com/langgraph"],
        }
    )

    assert critique["verdict"] == "need_improvement"
    assert any("autogen" in gap.lower() for gap in critique["gaps"])
    assert any("crewai" in gap.lower() for gap in critique["gaps"])
    assert critique["scores"]["coverage"] < 100
    assert critique["stats"]["comparative_points_count"] == 0
    assert any("横向对比不足" in gap for gap in critique["gaps"])
    assert any("限制条件" in gap for gap in critique["gaps"])


def test_critic_accepts_well_covered_notes() -> None:
    critique = _build_critique(
        {
            "summary": (
                "LangGraph focuses on workflow state orchestration, "
                "AutoGen focuses on conversation-driven collaboration, "
                "CrewAI focuses on role-based task execution."
            ),
            "key_points": [
                "LangGraph is strong for stateful workflow control and observability.",
                "AutoGen supports fast multi-agent experimentation with flexible interaction styles.",
                "CrewAI structures teams by role and is practical for business task flows.",
                "Comparison: LangGraph is more deterministic, AutoGen is more conversational, CrewAI is role-centric.",
                "For small teams, start with CrewAI or AutoGen, then migrate to LangGraph for stricter control.",
                "Trade-off: LangGraph has higher complexity and maintenance overhead, while CrewAI is easier to onboard for business workflows.",
            ],
            "references": [
                "https://langchain-ai.github.io/langgraph/",
                "https://microsoft.github.io/autogen/",
                "https://docs.crewai.com/",
            ],
        }
    )

    assert critique["verdict"] == "acceptable"
    assert critique["scores"]["overall"] >= 70
    assert "coverage" in critique["scores"]
    assert "reasoning_depth" in critique["scores"]
    assert critique["stats"]["source_domains_count"] >= 1
    assert critique["stats"]["comparative_points_count"] >= 1
    assert "dimension_coverage" in critique["scores"]
    assert "recommendations" in critique
    assert critique["verdict"] == "acceptable"


def test_critic_tracks_dimension_coverage_from_structured_notes() -> None:
    critique = _build_critique(
        {
            "summary": "Structured notes with partial dimension coverage.",
            "key_points": [
                "LangGraph vs AutoGen comparison with workflow focus.",
                "CrewAI role delegation for business processes.",
            ],
            "references": [
                "https://langchain-ai.github.io/langgraph/",
                "https://microsoft.github.io/autogen/",
            ],
            "structured": {
                "dimensions": {
                    "core_paradigm": {
                        "langgraph": ["LangGraph: graph paradigm."],
                        "autogen": ["AutoGen: conversation paradigm."],
                        "crewai": ["CrewAI: role-task paradigm."],
                        "common": [],
                    },
                    "coordination_style": {
                        "langgraph": ["LangGraph: node-edge coordination."],
                        "autogen": ["AutoGen: turn-based coordination."],
                        "crewai": ["CrewAI: delegation coordination."],
                        "common": [],
                    },
                    "state_memory": {
                        "langgraph": ["LangGraph: checkpointed state."],
                        "autogen": [],
                        "crewai": [],
                        "common": [],
                    },
                }
            },
        }
    )

    assert critique["scores"]["dimension_coverage"] < 100
    assert critique["stats"]["dimensions_covered_count"] == 3
    assert "best_fit" in critique["stats"]["dimensions_missing"]
    assert any("维度覆盖不足" in gap for gap in critique["gaps"])


class _StaticModel:
    def __init__(self, output: str) -> None:
        self.output = output

    def generate(self, messages: list[LLMMessage]) -> ModelResponse:
        del messages
        return ModelResponse(content=self.output, model_name="static")


def _build_context_with_notes(notes: dict[str, Any]) -> RuntimeContext:
    blackboard = Blackboard()
    blackboard.write("notes", notes, author="reader")
    return RuntimeContext(blackboard=blackboard)


def _critic_request() -> Message:
    return Message(sender="supervisor", receiver="critic", content="review notes")


def test_critic_llm_mode_uses_model_output_when_valid_json() -> None:
    model_output = """
    {
      "strengths": ["结构清晰"],
      "gaps": ["缺少成本评估"],
      "verdict": "acceptable",
      "scores": {
        "overall": 82,
        "coverage": 84,
        "evidence": 75,
        "specificity": 80,
        "balance": 78
      },
      "stats": {
        "key_points_count": 6,
        "references_count": 4,
        "missing_frameworks": []
      },
      "recommendations": ["补充量化指标"]
    }
    """
    agent = CriticAgent(model=_StaticModel(model_output), critic_mode="llm")
    context = _build_context_with_notes(
        {
            "summary": "LangGraph, AutoGen, CrewAI are all covered.",
            "key_points": ["LangGraph vs AutoGen vs CrewAI detailed comparison."],
            "references": ["https://example.com/1", "https://example.com/2"],
        }
    )

    response = agent.run(_critic_request(), context)
    critique = context.blackboard.read("critique", {}) if context.blackboard else {}

    assert response.type == "response"
    assert critique["assessment_mode"] == "llm"
    assert critique["verdict"] == "acceptable"
    assert critique["scores"]["overall"] == 82.0
    assert "rule_backup" in critique
    assert "stats" in critique["rule_backup"]


def test_critic_llm_mode_falls_back_to_rule_when_json_invalid() -> None:
    agent = CriticAgent(model=_StaticModel("not-json"), critic_mode="llm")
    context = _build_context_with_notes(
        {
            "summary": "Focus on LangGraph only.",
            "key_points": ["LangGraph has deterministic graph workflow."],
            "references": ["https://example.com/langgraph"],
        }
    )

    agent.run(_critic_request(), context)
    critique = context.blackboard.read("critique", {}) if context.blackboard else {}

    assert critique["assessment_mode"] == "rule_fallback"
    assert critique["model_fallback_reason"] == "critic_model_invalid_json"
    assert critique["verdict"] == "need_improvement"
    assert "rule_backup" not in critique


def test_critic_llm_mode_without_model_falls_back_to_rule() -> None:
    agent = CriticAgent(model=None, critic_mode="llm")
    context = _build_context_with_notes(
        {
            "summary": "Focus on LangGraph only.",
            "key_points": ["LangGraph has deterministic graph workflow."],
            "references": ["https://example.com/langgraph"],
        }
    )

    agent.run(_critic_request(), context)
    critique = context.blackboard.read("critique", {}) if context.blackboard else {}

    assert critique["assessment_mode"] == "rule_fallback"
    assert critique["model_fallback_reason"] == "critic_mode_llm_but_model_missing"


class _StaticCriticStrategy:
    def build_critique(self, request: CriticStrategyInput) -> dict[str, Any]:
        del request
        return {
            "strengths": ["custom strength"],
            "gaps": [],
            "verdict": "acceptable",
            "assessment_mode": "custom",
        }


def test_critic_agent_uses_custom_strategy() -> None:
    agent = CriticAgent(strategy=_StaticCriticStrategy(), critic_mode="rule")
    context = _build_context_with_notes(
        {
            "summary": "any",
            "key_points": ["any"],
            "references": ["https://example.com"],
        }
    )

    response = agent.run(_critic_request(), context)
    critique = context.blackboard.read("critique", {}) if context.blackboard else {}

    assert response.type == "response"
    assert critique["assessment_mode"] == "custom"
    assert critique["verdict"] == "acceptable"
