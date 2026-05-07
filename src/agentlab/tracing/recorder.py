from __future__ import annotations

import json
from pathlib import Path

from agentlab.core.event import Event


class TraceRecorder:
    def __init__(self) -> None:
        self._events: list[Event] = []

    def record(self, event: Event) -> None:
        self._events.append(event)

    def to_dict(self) -> dict[str, object]:
        return {
            "count": len(self._events),
            "events": [event.model_dump() for event in self._events],
        }

    def export_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target
