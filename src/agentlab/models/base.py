from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate text from a list of role/content messages."""
