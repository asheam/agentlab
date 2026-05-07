from typing import Any, Literal

from pydantic import BaseModel, Field


MessageType = Literal["task", "response", "tool_call", "tool_result", "error"]
MESSAGE_TYPES: set[str] = {"task", "response", "tool_call", "tool_result", "error"}


class Message(BaseModel):
    sender: str
    receiver: str
    content: str
    type: MessageType = "task"
    metadata: dict[str, Any] = Field(default_factory=dict)
