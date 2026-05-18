from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from agentlab.core.event import Event
from agentlab.tracing.recorder import TraceRecorder


def export_events_json(events: Sequence[Event], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [event.model_dump() for event in events]
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def export_trace_json(recorder: TraceRecorder, path: str | Path) -> Path:
    return recorder.export_json(path)


def export_trace_summary_json(recorder: TraceRecorder, path: str | Path) -> Path:
    return recorder.export_summary_json(path)
