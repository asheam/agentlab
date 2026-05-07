from __future__ import annotations

from pathlib import Path

from agentlab.agents.writer_agent import WriterAgent
from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message
from agentlab.tracing.recorder import TraceRecorder
from agentlab.workspace.artifacts import ArtifactStore
from agentlab.workspace.blackboard import Blackboard


class FixedModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, messages: list[dict[str, str]]) -> str:
        return self.response


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
