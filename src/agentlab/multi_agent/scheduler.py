from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol


AgentName = Literal["planner", "searcher", "reader", "critic", "writer", "supervisor", "runtime"]


class Scheduler(Protocol):
    def get_order(self) -> list[AgentName]:
        """Return the execution order for supervisor-worker flow."""


class FixedOrderScheduler:
    def __init__(self, order: Sequence[AgentName] | None = None) -> None:
        default_order: tuple[AgentName, ...] = (
            "planner",
            "searcher",
            "reader",
            "critic",
            "writer",
        )
        self._order = list(order) if order is not None else list(default_order)

    def get_order(self) -> list[AgentName]:
        return list(self._order)
