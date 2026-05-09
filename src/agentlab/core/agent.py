from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message
from agentlab.models.base import BaseModel
from agentlab.tools.base import BaseTool


ServiceName = Literal["runtime", "tool_registry", "blackboard", "trace_recorder", "artifacts"]


class Agent(ABC):
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        model: BaseModel | None = None,
        tools: Sequence[BaseTool] | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.tools = list(tools or [])

    @property
    def required_services(self) -> set[ServiceName]:
        return set()

    @abstractmethod
    def run(self, message: Message, context: RuntimeContext) -> Message:
        """Execute the agent logic for one message."""
