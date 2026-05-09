from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class BaseTool(ABC):
    name: str = ""
    description: str = ""

    @property
    def parameters_schema(self) -> Mapping[str, Any]:
        return {"type": "object", "properties": {}}

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Run the tool with keyword arguments."""
