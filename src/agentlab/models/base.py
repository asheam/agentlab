from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


LLMRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class LLMMessage:
    role: LLMRole
    content: str


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tokens_input: int = 0
    tokens_output: int = 0
    model_name: str = ""
    finish_reason: str = ""


class BaseModel(ABC):
    @abstractmethod
    def generate(self, messages: list[LLMMessage]) -> ModelResponse:
        """Generate text from typed chat messages."""
