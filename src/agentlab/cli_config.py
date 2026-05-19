from __future__ import annotations

from pathlib import Path
import re
from typing import Any


class CLIConfigError(ValueError):
    """Raised when CLI config file loading or validation fails."""


_ALLOWED_KEYS = {
    "topic",
    "use_openai",
    "output_dir",
    "search_mode",
    "no_search_fallback",
    "search_providers",
    "critic_mode",
    "strategy_preset",
    "max_retries",
    "agent_timeout_s",
    "retry_backoff_s",
    "retry_on_timeout_only",
    "continue_on_error",
}


def load_cli_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    file_path = Path(path)
    if not file_path.exists():
        raise CLIConfigError(f"Config file not found: {file_path}")

    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CLIConfigError(f"Failed to read config file '{file_path}': {exc}") from exc

    payload = _parse_yaml(text)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise CLIConfigError("Config root must be a mapping/object.")

    return _normalize_config(payload)


def _parse_yaml(text: str) -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)

    try:
        return yaml.safe_load(text)  # type: ignore[no-any-return,attr-defined]
    except Exception as exc:  # pragma: no cover - backend parser detail
        raise CLIConfigError(f"Invalid YAML config: {exc}") from exc


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    lines = text.splitlines()
    index = 0

    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        if raw.startswith((" ", "\t")):
            raise CLIConfigError("Nested mappings are not supported in simple YAML mode.")
        if ":" not in raw:
            raise CLIConfigError(f"Invalid line in config: {raw}")

        key_raw, value_raw = raw.split(":", 1)
        key = key_raw.strip()
        value = value_raw.strip()
        if not key:
            raise CLIConfigError(f"Invalid key in config line: {raw}")

        if value == "":
            list_items: list[Any] = []
            while index < len(lines):
                item_raw = lines[index]
                item_stripped = item_raw.strip()
                if not item_stripped or item_stripped.startswith("#"):
                    index += 1
                    continue
                if not item_raw.startswith((" ", "\t")):
                    break
                normalized = item_raw.lstrip()
                if not normalized.startswith("-"):
                    raise CLIConfigError(f"Expected list item under key '{key}': {item_raw}")
                list_items.append(_parse_scalar(normalized[1:].strip()))
                index += 1
            data[key] = list_items
            continue

        data[key] = _parse_scalar(value)

    return data


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if stripped.startswith(("'", '"')) and stripped.endswith(("'", '"')) and len(stripped) >= 2:
        quote = stripped[0]
        if stripped[-1] == quote:
            stripped = stripped[1:-1]

    lowered = stripped.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "~"}:
        return None

    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]

    if re.fullmatch(r"[+-]?\d+", stripped):
        try:
            return int(stripped)
        except ValueError:
            return stripped
    if re.fullmatch(r"[+-]?(?:\d+\.\d+|\d+\.)", stripped):
        try:
            return float(stripped)
        except ValueError:
            return stripped

    return stripped


def _normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(key for key in raw if key not in _ALLOWED_KEYS)
    if unknown:
        raise CLIConfigError(f"Unknown config keys: {', '.join(unknown)}")

    normalized: dict[str, Any] = {}
    if "topic" in raw:
        normalized["topic"] = _as_string(raw["topic"], "topic")
    if "use_openai" in raw:
        normalized["use_openai"] = _as_bool(raw["use_openai"], "use_openai")
    if "output_dir" in raw:
        normalized["output_dir"] = _as_string(raw["output_dir"], "output_dir")
    if "search_mode" in raw:
        value = _as_string(raw["search_mode"], "search_mode")
        if value not in {"mock", "real"}:
            raise CLIConfigError("search_mode must be 'mock' or 'real'.")
        normalized["search_mode"] = value
    if "no_search_fallback" in raw:
        normalized["no_search_fallback"] = _as_bool(
            raw["no_search_fallback"], "no_search_fallback"
        )
    if "search_providers" in raw:
        normalized["search_providers"] = _as_search_providers(raw["search_providers"])
    if "critic_mode" in raw:
        value = _as_string(raw["critic_mode"], "critic_mode")
        if value not in {"auto", "rule", "llm"}:
            raise CLIConfigError("critic_mode must be one of: auto, rule, llm.")
        normalized["critic_mode"] = value
    if "strategy_preset" in raw:
        value = _as_string(raw["strategy_preset"], "strategy_preset")
        if value not in {"default", "concise"}:
            raise CLIConfigError("strategy_preset must be one of: default, concise.")
        normalized["strategy_preset"] = value
    if "max_retries" in raw:
        normalized["max_retries"] = _as_non_negative_int(raw["max_retries"], "max_retries")
    if "agent_timeout_s" in raw:
        normalized["agent_timeout_s"] = _as_positive_float_or_none(
            raw["agent_timeout_s"], "agent_timeout_s"
        )
    if "retry_backoff_s" in raw:
        normalized["retry_backoff_s"] = _as_non_negative_float(
            raw["retry_backoff_s"], "retry_backoff_s"
        )
    if "retry_on_timeout_only" in raw:
        normalized["retry_on_timeout_only"] = _as_bool(
            raw["retry_on_timeout_only"], "retry_on_timeout_only"
        )
    if "continue_on_error" in raw:
        normalized["continue_on_error"] = _as_bool(
            raw["continue_on_error"], "continue_on_error"
        )

    return normalized


def _as_string(value: Any, key: str) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    raise CLIConfigError(f"Config key '{key}' must be a non-empty string.")


def _as_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise CLIConfigError(f"Config key '{key}' must be a boolean.")


def _as_search_providers(value: Any) -> list[str]:
    if isinstance(value, str):
        str_providers = [item.strip() for item in value.split(",") if item.strip()]
        if str_providers:
            return str_providers
        raise CLIConfigError("search_providers cannot be empty.")

    if isinstance(value, list):
        list_providers: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise CLIConfigError("search_providers list items must be strings.")
            cleaned = item.strip()
            if cleaned:
                list_providers.append(cleaned)
        if list_providers:
            return list_providers
        raise CLIConfigError("search_providers cannot be empty.")

    raise CLIConfigError("search_providers must be a comma-separated string or string list.")


def _as_non_negative_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise CLIConfigError(f"Config key '{key}' must be an integer >= 0.")
    if isinstance(value, int):
        if value >= 0:
            return value
    raise CLIConfigError(f"Config key '{key}' must be an integer >= 0.")


def _as_positive_float_or_none(value: Any, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise CLIConfigError(f"Config key '{key}' must be a number > 0 or null.")
    if isinstance(value, (int, float)):
        if float(value) > 0:
            return float(value)
    raise CLIConfigError(f"Config key '{key}' must be a number > 0 or null.")


def _as_non_negative_float(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise CLIConfigError(f"Config key '{key}' must be a number >= 0.")
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric >= 0:
            return numeric
    raise CLIConfigError(f"Config key '{key}' must be a number >= 0.")
