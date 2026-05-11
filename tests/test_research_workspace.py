from __future__ import annotations

from agentlab.workspace.blackboard import Blackboard
from agentlab.workspace.research_workspace import (
    read_critique,
    read_notes,
    read_plan,
    read_report,
    read_search_results,
    write_critique,
    write_notes,
    write_plan,
    write_report,
    write_search_results,
)


def test_research_workspace_read_defaults_for_missing_keys() -> None:
    board = Blackboard()

    assert read_plan(board) == []
    assert read_search_results(board) == []
    assert read_notes(board) == {"key_points": [], "references": [], "summary": "", "structured": {}}
    assert read_critique(board) == {}
    assert read_report(board) == ""


def test_research_workspace_read_coerces_invalid_types() -> None:
    board = Blackboard()
    board.write("plan", {"bad": "type"}, author="planner")
    board.write("search_results", "invalid", author="searcher")
    board.write("notes", "invalid", author="reader")
    board.write("critique", "invalid", author="critic")
    board.write("report", 12345, author="writer")

    assert read_plan(board) == []
    assert read_search_results(board) == []
    assert read_notes(board) == {"key_points": [], "references": [], "summary": "", "structured": {}}
    assert read_critique(board) == {}
    assert read_report(board) == "12345"


def test_research_workspace_write_read_roundtrip() -> None:
    board = Blackboard()

    write_plan(board, ["q1", "q2"], author="planner")
    write_search_results(board, [{"question": "q1", "result": {"results": []}}], author="searcher")
    write_notes(
        board,
        {
            "key_points": ["k1"],
            "references": ["mock://a"],
            "summary": "sum",
            "structured": {"langgraph": {"points": ["k1"], "references": ["mock://a"]}},
        },
        author="reader",
    )
    write_critique(
        board,
        {
            "verdict": "acceptable",
            "strengths": ["s1"],
            "scores": {"overall": 80},
        },
        author="critic",
    )
    write_report(board, "# Report", author="writer")

    assert read_plan(board) == ["q1", "q2"]
    assert read_search_results(board)[0]["question"] == "q1"
    assert read_notes(board)["summary"] == "sum"
    assert read_critique(board)["verdict"] == "acceptable"
    assert read_report(board) == "# Report"
