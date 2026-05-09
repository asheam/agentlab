from __future__ import annotations

from agentlab.agents.critic_agent import _build_critique


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
            ],
            "references": [
                "https://example.com/langgraph",
                "https://example.com/autogen",
                "https://example.com/crewai",
            ],
        }
    )

    assert critique["verdict"] == "acceptable"
    assert critique["scores"]["overall"] >= 70
    assert "coverage" in critique["scores"]
    assert "recommendations" in critique
    assert any("覆盖完整度良好" in gap for gap in critique["gaps"])
