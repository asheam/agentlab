# AgentLab v0.6.0 Release Notes (Draft)

Release date: 2026-05-19

## Goal

v0.6.0 focuses on runtime observability and retry-policy control:

- Structured search-provider telemetry (hits/errors/fallback)
- Configurable retry behavior (backoff, timeout-only retries)
- Expanded `run_summary.json` metrics

## Highlights

1. Structured search telemetry

- `web_search` now emits `provider_errors` in both real and fallback paths.
- `SearchAgent` tool-call event metadata now includes:
  - `search_mode`
  - `fallback_used`
  - `source_hits`
  - `provider_errors`
- `Writer` now prefers structured provider error counters when building Search Mode Summary, and falls back to string parsing only when structured counters are absent.

2. RunPolicy enhancements

`RunPolicy` now supports:

- `retry_backoff_s: float = 0.0`
- `retry_on_timeout_only: bool = False`

Behavior:

- If `retry_backoff_s > 0`, supervisor sleeps before retry.
- If `retry_on_timeout_only=True`, only timeout failures are retried; runtime errors are not retried.

3. `run_summary.json` expansion

New summary blocks:

- `search_stats` (`queries`, `mode_counts`, `provider_hits`, `provider_errors`, `fallback_used`)
- `retry_stats` (`total_retries`, `timeout_retries`, `error_retries`)

## Compatibility

- No breaking API change.
- Existing CLI commands, examples, and output file paths remain compatible.

## Validation (draft)

- `pytest` passed
- `ruff check .` passed
- `mypy src` passed
- `uv run agentlab "Research LangGraph AutoGen CrewAI differences" --search-mode real --no-search-fallback` passed