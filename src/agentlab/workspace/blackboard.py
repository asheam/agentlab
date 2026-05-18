from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BlackboardEntry(BaseModel):
    key: str
    value: Any
    author: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Blackboard:
    def __init__(self) -> None:
        self._entries: dict[str, BlackboardEntry] = {}

    def write(self, key: str, value: Any, author: str) -> BlackboardEntry:
        entry = BlackboardEntry(key=key, value=value, author=author)
        self._entries[key] = entry
        return entry

    def read(self, key: str, default: Any = None) -> Any:
        entry = self._entries.get(key)
        if entry is None:
            return default
        return entry.value

    def list(self) -> list[str]:
        return list(self._entries.keys())

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {key: entry.model_dump() for key, entry in self._entries.items()}

    def export_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            __import__("json").dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target
