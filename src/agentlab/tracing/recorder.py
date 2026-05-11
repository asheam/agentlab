from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agentlab.core.event import Event


@dataclass
class _AgentSummary:
    events: int = 0
    success: int = 0
    failure: int = 0
    latency_ms_total: float = 0.0

    def add(self, success: bool, latency_ms: float | None) -> None:
        self.events += 1
        if success:
            self.success += 1
        else:
            self.failure += 1
        if latency_ms is not None:
            self.latency_ms_total += latency_ms

    def to_dict(self) -> dict[str, float | int]:
        avg = self.latency_ms_total / self.events if self.events > 0 else 0.0
        return {
            "events": self.events,
            "success": self.success,
            "failure": self.failure,
            "latency_ms_total": round(self.latency_ms_total, 3),
            "latency_ms_avg": round(avg, 3),
        }


@dataclass
class _ToolSummary:
    calls: int = 0
    success: int = 0
    failure: int = 0
    latency_ms_total: float = 0.0

    def add(self, success: bool, latency_ms: float | None) -> None:
        self.calls += 1
        if success:
            self.success += 1
        else:
            self.failure += 1
        if latency_ms is not None:
            self.latency_ms_total += latency_ms

    def to_dict(self) -> dict[str, float | int]:
        avg = self.latency_ms_total / self.calls if self.calls > 0 else 0.0
        return {
            "calls": self.calls,
            "success": self.success,
            "failure": self.failure,
            "latency_ms_total": round(self.latency_ms_total, 3),
            "latency_ms_avg": round(avg, 3),
        }


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

    def to_summary_dict(self) -> dict[str, object]:
        event_type_counts: dict[str, int] = {}
        agent_stats: dict[str, _AgentSummary] = {}
        tool_stats: dict[str, _ToolSummary] = {}
        latencies: list[float] = []
        failures = 0

        for event in self._events:
            event_type = event.event_type
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            if not event.success:
                failures += 1

            if event.latency_ms is not None:
                latencies.append(event.latency_ms)

            if event.agent:
                agent_summary = agent_stats.setdefault(event.agent, _AgentSummary())
                agent_summary.add(success=event.success, latency_ms=event.latency_ms)

            if event.tool_name:
                tool_summary = tool_stats.setdefault(event.tool_name, _ToolSummary())
                tool_summary.add(success=event.success, latency_ms=event.latency_ms)

        total_latency_ms = sum(latencies)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_events": len(self._events),
            "success_events": len(self._events) - failures,
            "failed_events": failures,
            "event_type_counts": event_type_counts,
            "latency": {
                "total_ms": round(total_latency_ms, 3),
                "average_ms": round(total_latency_ms / len(latencies), 3) if latencies else 0.0,
                "max_ms": round(max(latencies), 3) if latencies else 0.0,
            },
            "agent_stats": {name: stats.to_dict() for name, stats in agent_stats.items()},
            "tool_stats": {name: stats.to_dict() for name, stats in tool_stats.items()},
        }

    def export_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def export_summary_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_summary_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target
