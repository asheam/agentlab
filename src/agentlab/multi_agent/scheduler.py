from __future__ import annotations


class FixedOrderScheduler:
    def __init__(self, order: list[str] | None = None) -> None:
        self._order = order or ["planner", "searcher", "reader", "critic", "writer"]

    def get_order(self) -> list[str]:
        return list(self._order)
