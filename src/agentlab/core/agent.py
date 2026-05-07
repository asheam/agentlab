from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message


class Agent(ABC):
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        model: Any = None,
        tools: Sequence[Any] | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.tools = list(tools or [])

    @abstractmethod
    def run(self, message: Message, context: RuntimeContext) -> Message:
        """Execute the agent logic for one message."""
