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
        search_mode_counts: dict[str, int] = {}
        search_provider_hits = {"duckduckgo": 0, "wikipedia": 0, "tavily": 0}
        search_provider_errors = {"duckduckgo": 0, "wikipedia": 0, "tavily": 0}
        search_queries = 0
        search_fallback_used = 0
        retry_total = 0
        retry_timeout = 0
        retry_runtime_error = 0
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
                metadata = event.metadata
                if (
                    event.event_type == "agent_call"
                    and metadata.get("emitter") == "supervisor"
                    and metadata.get("will_retry") is True
                ):
                    retry_total += 1
                    reason = metadata.get("reason")
                    if reason == "timeout":
                        retry_timeout += 1
                    elif reason == "runtime_error":
                        retry_runtime_error += 1

            if event.tool_name:
                tool_summary = tool_stats.setdefault(event.tool_name, _ToolSummary())
                tool_summary.add(success=event.success, latency_ms=event.latency_ms)
                if event.tool_name == "web_search":
                    search_queries += 1
                    metadata = event.metadata
                    mode = metadata.get("search_mode")
                    if isinstance(mode, str) and mode:
                        search_mode_counts[mode] = search_mode_counts.get(mode, 0) + 1
                    if metadata.get("fallback_used") is True:
                        search_fallback_used += 1
                    _increment_provider_counter(
                        search_provider_hits,
                        metadata.get("source_hits"),
                    )
                    metadata_provider_errors = metadata.get("provider_errors")
                    _increment_provider_counter(
                        search_provider_errors,
                        metadata_provider_errors,
                    )
                    if event.error and not _has_nonzero_provider_counts(metadata_provider_errors):
                        _increment_provider_counter(
                            search_provider_errors,
                            _provider_errors_from_text(event.error),
                        )

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
            "search_stats": {
                "queries": search_queries,
                "fallback_used": search_fallback_used,
                "mode_counts": search_mode_counts,
                "provider_hits": search_provider_hits,
                "provider_errors": search_provider_errors,
            },
            "retry_stats": {
                "total_retries": retry_total,
                "timeout_retries": retry_timeout,
                "error_retries": retry_runtime_error,
            },
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


def _increment_provider_counter(target: dict[str, int], raw: object) -> None:
    if not isinstance(raw, dict):
        return
    for provider in target:
        value = raw.get(provider)
        if value is None:
            continue
        target[provider] += _to_int(value)


def _provider_errors_from_text(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {
        "duckduckgo": lowered.count("duckduckgo_error:"),
        "wikipedia": lowered.count("wikipedia_error:"),
        "tavily": lowered.count("tavily_error:"),
    }


def _has_nonzero_provider_counts(raw: object) -> bool:
    if not isinstance(raw, dict):
        return False
    for provider in ("duckduckgo", "wikipedia", "tavily"):
        value = raw.get(provider)
        if value is None:
            continue
        if _to_int(value) > 0:
            return True
    return False


def _to_int(raw: object) -> int:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return 0
        try:
            return int(float(text))
        except ValueError:
            return 0
    return 0
