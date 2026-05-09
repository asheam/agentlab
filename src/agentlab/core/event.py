from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


EventType = Literal["agent_call", "tool_call", "model_call"]


class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: EventType
    agent: str | None = None
    input: str | None = None
    output: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    latency_ms: float | None = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
