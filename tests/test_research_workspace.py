from __future__ import annotations

from agentlab.workspace.blackboard import Blackboard
from agentlab.workspace.research_workspace import (
    critique_assessment_mode,
    critique_gaps,
    critique_recommendations,
    critique_scores,
    critique_stats,
    critique_strengths,
    critique_verdict,
    notes_dimension_points,
    notes_framework_points,
    notes_key_points,
    notes_references,
    notes_summary,
    read_critique,
    read_notes,
    read_plan,
    read_report,
    read_search_results,
    search_result_error,
    search_result_fallback_reason,
    search_result_fallback_used,
    search_result_issues,
    search_result_mode,
    search_result_payload,
    search_result_provider_errors,
    search_result_source_hits,
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


def test_research_workspace_note_helpers_normalize_access() -> None:
    board = Blackboard()
    write_notes(
        board,
        {
            "key_points": [" p1 ", "", "p2"],
            "references": [" https://example.com ", " "],
            "summary": "summary",
            "structured": {
                "langgraph": {
                    "points": ["LG point", " "],
                    "references": ["mock://lg"],
                },
                "dimensions": {
                    "core_paradigm": {
                        "langgraph": ["LG core"],
                        "autogen": ["AG core"],
                    }
                },
            },
        },
        author="reader",
    )

    notes = read_notes(board)
    assert notes_key_points(notes) == ["p1", "p2"]
    assert notes_references(notes) == ["https://example.com"]
    assert notes_summary(notes) == "summary"
    assert notes_framework_points(notes, "langgraph") == ["LG point"]
    assert notes_dimension_points(notes, "core_paradigm", "autogen") == ["AG core"]
    assert notes_dimension_points(notes, "trade_off", "autogen") == []


def test_research_workspace_critique_helpers_and_search_helpers() -> None:
    board = Blackboard()
    write_critique(
        board,
        {
            "verdict": "acceptable",
            "assessment_mode": "rule",
            "strengths": ["s1"],
            "gaps": ["g1"],
            "recommendations": ["r1"],
            "scores": {"overall": 90},
            "stats": {"dimensions_covered_count": 3},
        },
        author="critic",
    )
    write_search_results(
        board,
        [
            {
                "question": "q1",
                "error": "wikipedia_error: blocked",
            },
            {
                "question": "q2",
                "result": {
                    "mode": "real",
                    "fallback_used": True,
                    "fallback_reason": "tavily_error: missing",
                    "source_hits": {"duckduckgo": "1", "wikipedia": 2, "tavily": 0},
                    "provider_errors": {"duckduckgo": 1, "wikipedia": "0", "tavily": 2},
                    "real_issues": ["duckduckgo_error: timeout"],
                },
            },
        ],
        author="searcher",
    )

    critique = read_critique(board)
    assert critique_verdict(critique) == "acceptable"
    assert critique_assessment_mode(critique) == "rule"
    assert critique_strengths(critique) == ["s1"]
    assert critique_gaps(critique) == ["g1"]
    assert critique_recommendations(critique) == ["r1"]
    assert critique_scores(critique)["overall"] == 90
    assert critique_stats(critique)["dimensions_covered_count"] == 3

    search_results = read_search_results(board)
    first_error = search_result_error(search_results[0])
    assert first_error == "wikipedia_error: blocked"

    payload = search_result_payload(search_results[1])
    assert search_result_mode(payload) == "real"
    assert search_result_fallback_used(payload) is True
    assert search_result_fallback_reason(payload) == "tavily_error: missing"
    assert search_result_source_hits(payload) == {"duckduckgo": 1, "wikipedia": 2, "tavily": 0}
    assert search_result_provider_errors(payload) == {"duckduckgo": 1, "wikipedia": 0, "tavily": 2}
    assert search_result_issues(payload) == ["duckduckgo_error: timeout"]
