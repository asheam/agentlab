from __future__ import annotations

from agentlab.agents.reader_agent import ReaderAgent, ReaderStrategyInput, _build_notes
from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message
from agentlab.workspace.blackboard import Blackboard
from agentlab.workspace.research_workspace import NotesPayload


def _sample_search_results() -> list[dict[str, object]]:
    return [
        {
            "question": "compare frameworks",
            "result": {
                "results": [
                    {
                        "title": "LangGraph overview",
                        "snippet": "Uses graph-based orchestration with state checkpoints.",
                        "source": "mock://langgraph/overview",
                    },
                    {
                        "title": "AutoGen overview",
                        "snippet": "Conversation loop with message turn-taking.",
                        "source": "mock://autogen/overview",
                    },
                    {
                        "title": "CrewAI overview",
                        "snippet": "Role and task delegation for business workflows.",
                        "source": "mock://crewai/overview",
                    },
                    {
                        "title": "Framework comparison snapshot",
                        "snippet": "LangGraph vs AutoGen vs CrewAI trade-off summary.",
                        "source": "mock://comparison/langgraph-autogen-crewai",
                    },
                ]
            },
        }
    ]


def test_build_notes_includes_structured_framework_buckets() -> None:
    notes = _build_notes(_sample_search_results())

    assert "structured" in notes
    structured = notes["structured"]
    assert isinstance(structured, dict)
    assert "langgraph" in structured
    assert "autogen" in structured
    assert "crewai" in structured
    assert "common" in structured
    assert "dimensions" in structured

    langgraph_points = structured["langgraph"]["points"]
    autogen_points = structured["autogen"]["points"]
    crewai_points = structured["crewai"]["points"]
    common_points = structured["common"]["points"]
    dimensions = structured["dimensions"]

    assert any(point.startswith("LangGraph:") for point in langgraph_points)
    assert any(point.startswith("AutoGen:") for point in autogen_points)
    assert any(point.startswith("CrewAI:") for point in crewai_points)
    assert any("LangGraph vs AutoGen vs CrewAI" in point for point in common_points)
    assert isinstance(dimensions, dict)
    assert "coordination_style" in dimensions
    assert "trade_off" in dimensions
    assert any(
        dimensions[dimension]["langgraph"] for dimension in dimensions if isinstance(dimensions[dimension], dict)
    )
    assert any(dimensions["trade_off"]["common"])

    assert len(notes["references"]) == 4
    assert len(notes["key_points"]) >= 4


def test_reader_agent_writes_structured_notes_to_blackboard() -> None:
    blackboard = Blackboard()
    blackboard.write("search_results", _sample_search_results(), author="searcher")
    context = RuntimeContext(blackboard=blackboard)
    agent = ReaderAgent()

    response = agent.run(
        Message(sender="supervisor", receiver="reader", content="summarize", type="task"),
        context,
    )

    assert response.type == "response"
    saved_notes = blackboard.read("notes")
    assert isinstance(saved_notes, dict)
    assert "structured" in saved_notes
    assert isinstance(saved_notes["structured"], dict)
    assert "dimensions" in saved_notes["structured"]


class _StaticReaderStrategy:
    def build_notes(self, request: ReaderStrategyInput) -> NotesPayload:
        del request
        return NotesPayload(
            key_points=["custom point"],
            references=["mock://custom/ref"],
            summary="custom summary",
            structured={},
        )


def test_reader_agent_uses_custom_strategy() -> None:
    blackboard = Blackboard()
    blackboard.write("search_results", _sample_search_results(), author="searcher")
    context = RuntimeContext(blackboard=blackboard)
    agent = ReaderAgent(strategy=_StaticReaderStrategy())

    response = agent.run(
        Message(sender="supervisor", receiver="reader", content="summarize", type="task"),
        context,
    )

    assert response.type == "response"
    saved_notes = blackboard.read("notes")
    assert saved_notes["summary"] == "custom summary"
    assert saved_notes["key_points"] == ["custom point"]


def test_reader_dimension_prefers_question_templates_and_deduplicates() -> None:
    notes = _build_notes(
        [
            {
                "question": "它们各自的核心设计理念和目标场景是什么？",
                "result": {
                    "results": [
                        {
                            "title": "LangGraph core",
                            "snippet": "Uses graph-based orchestration with explicit state machine flow.",
                            "source": "mock://langgraph/core",
                        }
                    ]
                },
            },
            {
                "question": "任务编排方式（图、对话、角色分工）有哪些关键差异？",
                "result": {
                    "results": [
                        {
                            "title": "LangGraph orchestration",
                            "snippet": "Uses graph-based orchestration with explicit state machine flow.",
                            "source": "mock://langgraph/core",
                        },
                        {
                            "title": "AutoGen orchestration",
                            "snippet": "Conversation turn-based coordination and multi-agent message loop.",
                            "source": "mock://autogen/orchestration",
                        },
                    ]
                },
            },
        ]
    )

    structured = notes["structured"]
    dimensions = structured["dimensions"]
    core_langgraph = dimensions["core_paradigm"]["langgraph"]
    coordination_langgraph = dimensions["coordination_style"]["langgraph"]
    coordination_autogen = dimensions["coordination_style"]["autogen"]

    assert any("LangGraph:" in point for point in core_langgraph)
    assert coordination_langgraph == []
    assert any("AutoGen:" in point for point in coordination_autogen)
