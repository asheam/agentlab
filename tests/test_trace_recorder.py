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
