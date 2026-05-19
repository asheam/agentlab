import json

from agentlab.core.event import Event
from agentlab.tracing.recorder import TraceRecorder


def test_trace_recorder_record_and_to_dict() -> None:
    recorder = TraceRecorder()
    recorder.record(
        Event(event_type="agent_call", agent="planner", input="topic", output="plan")
    )

    payload = recorder.to_dict()
    assert payload["count"] == 1
    assert payload["events"][0]["agent"] == "planner"


def test_trace_recorder_export_json(tmp_path) -> None:
    recorder = TraceRecorder()
    recorder.record(Event(event_type="tool_call", tool_name="calculator", success=True))

    out_path = tmp_path / "trace.json"
    recorder.export_json(out_path)

    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["events"][0]["tool_name"] == "calculator"


def test_trace_recorder_summary_and_export(tmp_path) -> None:
    recorder = TraceRecorder()
    recorder.record(
        Event(event_type="agent_call", agent="planner", success=True, latency_ms=10.0)
    )
    recorder.record(
        Event(
            event_type="tool_call",
            agent="searcher",
            tool_name="web_search",
            success=False,
            latency_ms=25.0,
            error="network error",
            metadata={
                "search_mode": "real",
                "fallback_used": True,
                "source_hits": {"duckduckgo": 0, "wikipedia": 1, "tavily": 0},
                "provider_errors": {"duckduckgo": 1, "wikipedia": 0, "tavily": 0},
            },
        )
    )

    summary = recorder.to_summary_dict()
    assert summary["total_events"] == 2
    assert summary["failed_events"] == 1
    assert summary["event_type_counts"]["agent_call"] == 1
    assert summary["event_type_counts"]["tool_call"] == 1
    assert summary["agent_stats"]["planner"]["events"] == 1
    assert summary["tool_stats"]["web_search"]["calls"] == 1
    assert summary["search_stats"]["queries"] == 1
    assert summary["search_stats"]["fallback_used"] == 1
    assert summary["search_stats"]["mode_counts"]["real"] == 1
    assert summary["search_stats"]["provider_hits"]["wikipedia"] == 1
    assert summary["search_stats"]["provider_errors"]["duckduckgo"] == 1
    assert summary["retry_stats"]["total_retries"] == 0
    assert summary["retry_stats"]["timeout_retries"] == 0
    assert summary["retry_stats"]["error_retries"] == 0

    out_path = tmp_path / "run_summary.json"
    recorder.export_summary_json(out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["total_events"] == 2
    assert "latency" in payload
