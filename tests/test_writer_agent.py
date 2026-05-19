from __future__ import annotations

from pathlib import Path

from agentlab.agents.writer_agent import (
    WriterAgent,
    WriterStrategyInput,
    _extract_framework_clause,
    _select_framework_cell,
)
from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message
from agentlab.models.base import LLMMessage, ModelResponse
from agentlab.tracing.recorder import TraceRecorder
from agentlab.workspace.artifacts import ArtifactStore
from agentlab.workspace.blackboard import Blackboard


class FixedModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, messages: list[LLMMessage]) -> ModelResponse:
        del messages
        return ModelResponse(content=self.response, model_name="fixed")


class StaticWriterStrategy:
    def build_report(self, request: WriterStrategyInput) -> str:
        del request
        return "# Deep Research Report\n\n## Topic\ncustom strategy\n"


def _build_context(tmp_path: Path) -> RuntimeContext:
    board = Blackboard()
    board.write("plan", ["Q1", "Q2"], author="planner")
    board.write("search_results", [{"question": "Q1", "result": {"results": []}}], author="searcher")
    board.write(
        "notes",
        {
            "key_points": ["LangGraph point", "AutoGen point", "CrewAI point"],
            "references": ["mock://langgraph/overview"],
            "summary": "summary",
        },
        author="reader",
    )
    board.write(
        "critique",
        {"strengths": ["s1"], "gaps": ["g1"], "verdict": "acceptable"},
        author="critic",
    )
    return RuntimeContext(
        blackboard=board,
        artifacts=ArtifactStore(base_dir=tmp_path),
        trace_recorder=TraceRecorder(),
    )


def test_writer_agent_uses_model_report_when_available(tmp_path: Path) -> None:
    model = FixedModel(
        "\n".join(
            [
                "# Deep Research Report",
                "",
                "## Topic",
                "topic",
                "",
                "## Research Questions",
                "- q1",
                "",
                "## Key Findings",
                "- k1",
                "",
                "## Framework Snapshot",
                "- f1",
                "",
                "## Comparison Matrix",
                "| Dimension | LangGraph | AutoGen | CrewAI |",
                "| --- | --- | --- | --- |",
                "| Core Paradigm | a | b | c |",
                "",
                "## Critique",
                "- c1",
                "",
                "## References",
                "- r1",
                "",
            ]
        )
    )
    agent = WriterAgent(model=model)
    context = _build_context(tmp_path)

    response = agent.run(
        Message(sender="supervisor", receiver="writer", content="topic", type="task", metadata={"topic": "topic"}),
        context,
    )

    assert response.content.startswith("# Deep Research Report")
    assert context.blackboard is not None
    assert context.blackboard.read("report") == response.content
    saved = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert saved.startswith("# Deep Research Report")
    assert context.trace_recorder is not None
    payload = context.trace_recorder.to_dict()
    assert payload["events"][-1]["event_type"] == "model_call"
    assert payload["events"][-1]["success"] is True


def test_writer_agent_uses_custom_strategy(tmp_path: Path) -> None:
    agent = WriterAgent(strategy=StaticWriterStrategy())
    context = _build_context(tmp_path)

    response = agent.run(
        Message(sender="supervisor", receiver="writer", content="topic", type="task", metadata={"topic": "topic"}),
        context,
    )

    assert response.content == "# Deep Research Report\n\n## Topic\ncustom strategy\n"
    assert context.blackboard is not None
    assert context.blackboard.read("report") == response.content


def test_writer_agent_falls_back_when_model_returns_empty(tmp_path: Path) -> None:
    model = FixedModel("  ")
    agent = WriterAgent(model=model)
    context = _build_context(tmp_path)

    response = agent.run(
        Message(sender="supervisor", receiver="writer", content="topic", type="task", metadata={"topic": "topic"}),
        context,
    )

    assert response.content.startswith("# Deep Research Report")
    assert "## Model Fallback" in response.content
    assert "## Comparison Matrix" in response.content
    assert context.trace_recorder is not None
    payload = context.trace_recorder.to_dict()
    assert payload["events"][-1]["event_type"] == "model_call"
    assert payload["events"][-1]["success"] is False


def test_writer_agent_falls_back_when_model_output_missing_headers(tmp_path: Path) -> None:
    model = FixedModel("# Deep Research Report\n\nOnly one section.")
    agent = WriterAgent(model=model)
    context = _build_context(tmp_path)

    response = agent.run(
        Message(sender="supervisor", receiver="writer", content="topic", type="task", metadata={"topic": "topic"}),
        context,
    )

    assert response.content.startswith("# Deep Research Report")
    assert "## Model Fallback" in response.content
    assert "## Comparison Matrix" in response.content


def test_writer_agent_reports_provider_error_counts(tmp_path: Path) -> None:
    agent = WriterAgent()
    context = _build_context(tmp_path)

    assert context.blackboard is not None
    context.blackboard.write(
        "search_results",
        [
            {
                "question": "Q1",
                "result": {
                    "mode": "real",
                    "fallback_used": False,
                    "source_hits": {"duckduckgo": 0, "wikipedia": 2, "tavily": 0},
                    "real_issues": [
                        "tavily_error: tavily_missing_api_key",
                        "duckduckgo_error: timeout",
                    ],
                },
            },
            {
                "question": "Q2",
                "result": {
                    "mode": "mock",
                    "fallback_used": True,
                    "source_hits": {"duckduckgo": 0, "wikipedia": 0, "tavily": 0},
                    "fallback_reason": "candidate A: tavily_error: tavily_missing_api_key",
                },
            },
            {
                "question": "Q3",
                "error": "Failed to execute tool 'web_search': wikipedia_error: blocked",
            },
        ],
        author="searcher",
    )

    response = agent.run(
        Message(sender="supervisor", receiver="writer", content="topic", type="task", metadata={"topic": "topic"}),
        context,
    )

    assert "## Comparison Matrix" in response.content
    assert "| Dimension | LangGraph | AutoGen | CrewAI |" in response.content
    assert "| DuckDuckGo errors | 1 |" in response.content
    assert "| Wikipedia errors | 1 |" in response.content
    assert "| Tavily errors | 2 |" in response.content
    assert "_Italic cells indicate fallback defaults when direct evidence is unavailable._" in response.content


def test_writer_agent_prefers_structured_provider_error_counts(tmp_path: Path) -> None:
    agent = WriterAgent()
    context = _build_context(tmp_path)

    assert context.blackboard is not None
    context.blackboard.write(
        "search_results",
        [
            {
                "question": "Q1",
                "result": {
                    "mode": "real",
                    "fallback_used": True,
                    "source_hits": {"duckduckgo": 0, "wikipedia": 0, "tavily": 0},
                    "provider_errors": {"duckduckgo": 2, "wikipedia": 3, "tavily": 4},
                    "real_issues": [
                        "duckduckgo_error: should_not_double_count",
                        "wikipedia_error: should_not_double_count",
                        "tavily_error: should_not_double_count",
                    ],
                },
            }
        ],
        author="searcher",
    )

    response = agent.run(
        Message(sender="supervisor", receiver="writer", content="topic", type="task", metadata={"topic": "topic"}),
        context,
    )

    assert "| DuckDuckGo errors | 2 |" in response.content
    assert "| Wikipedia errors | 3 |" in response.content
    assert "| Tavily errors | 4 |" in response.content


def test_extract_framework_clause_strips_natural_prefix() -> None:
    result = _extract_framework_clause(
        "LangGraph uses graph-based orchestration with state checkpoints.",
        "langgraph",
    )
    assert result == "uses graph-based orchestration with state checkpoints."


def test_select_framework_cell_marks_fallback_with_italics() -> None:
    cell = _select_framework_cell(
        framework="autogen",
        points=[],
        keywords=["conversation", "message"],
        fallback="Conversation-loop based multi-agent collaboration.",
    )
    assert cell.startswith("_")
    assert cell.endswith("_")


def test_select_framework_cell_uses_highest_keyword_match() -> None:
    cell = _select_framework_cell(
        framework="langgraph",
        points=[
            "LangGraph handles workflow.",
            "LangGraph has deterministic node-edge workflow with explicit state control.",
        ],
        keywords=["deterministic", "node", "edge", "state", "workflow"],
        fallback="Graph/state-machine orchestration with explicit transitions.",
    )
    assert "deterministic node-edge workflow" in cell


def test_writer_matrix_prefers_structured_notes(tmp_path: Path) -> None:
    agent = WriterAgent()
    context = _build_context(tmp_path)
    assert context.blackboard is not None

    context.blackboard.write(
        "notes",
        {
            "key_points": ["generic finding without framework label"],
            "references": ["mock://general/result"],
            "summary": "summary",
            "structured": {
                "langgraph": {
                    "points": ["LangGraph: framework-level generic note."],
                    "references": ["mock://langgraph/overview"],
                },
                "autogen": {
                    "points": ["AutoGen: framework-level generic note."],
                    "references": ["mock://autogen/overview"],
                },
                "crewai": {
                    "points": ["CrewAI: framework-level generic note."],
                    "references": ["mock://crewai/overview"],
                },
                "common": {"points": [], "references": []},
                "dimensions": {
                    "core_paradigm": {
                        "langgraph": ["LangGraph: graph/state-machine orchestration."],
                        "autogen": ["AutoGen: conversation loop orchestration."],
                        "crewai": ["CrewAI: role-task paradigm."],
                        "common": [],
                    },
                    "coordination_style": {
                        "langgraph": ["LangGraph: deterministic node-edge routing."],
                        "autogen": ["AutoGen: turn-based dialogue coordination."],
                        "crewai": ["CrewAI: delegation-based coordination."],
                        "common": [],
                    },
                    "state_memory": {
                        "langgraph": ["LangGraph: explicit checkpointed state."],
                        "autogen": ["AutoGen: chat-history centric context."],
                        "crewai": ["CrewAI: lightweight shared context."],
                        "common": [],
                    },
                    "best_fit": {
                        "langgraph": ["LangGraph: fit for complex audited workflows."],
                        "autogen": ["AutoGen: fit for rapid conversational prototyping."],
                        "crewai": ["CrewAI: fit for business process automation."],
                        "common": [],
                    },
                    "trade_off": {
                        "langgraph": ["LangGraph: higher setup and maintenance cost."],
                        "autogen": ["AutoGen: easy drift without strict constraints."],
                        "crewai": ["CrewAI: limited deep customization."],
                        "common": [],
                    },
                },
            },
        },
        author="reader",
    )

    response = agent.run(
        Message(sender="supervisor", receiver="writer", content="topic", type="task", metadata={"topic": "topic"}),
        context,
    )

    assert "## Comparison Matrix" in response.content
    assert "deterministic node-edge routing" in response.content
    assert "turn-based dialogue coordination" in response.content
    assert "delegation-based coordination" in response.content
