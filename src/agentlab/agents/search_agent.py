from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from agentlab.core.agent import Agent, ServiceName
from agentlab.core.context import RuntimeContext
from agentlab.core.event import Event
from agentlab.core.message import Message
from agentlab.models.base import BaseModel
from agentlab.workspace.research_workspace import (
    SearchResultItem,
    read_plan,
    write_search_results,
)


@dataclass(frozen=True)
class SearchStrategyInput:
    topic: str
    plan: list[str]
    tool_name: str
    context: RuntimeContext
    agent_name: str


@dataclass(frozen=True)
class SearchStrategyOutput:
    search_results: list[SearchResultItem]
    tool_errors: list[str]


class SearchStrategy(Protocol):
    def collect(self, request: SearchStrategyInput) -> SearchStrategyOutput:
        """Collect search results for all planned questions."""


class DefaultSearchStrategy:
    def collect(self, request: SearchStrategyInput) -> SearchStrategyOutput:
        if request.context.tool_registry is None:
            raise RuntimeError("tool_registry is required for SearchAgent")

        search_results: list[SearchResultItem] = []
        tool_errors: list[str] = []
        for question in request.plan:
            query = f"{request.topic} {question}".strip() if request.topic else question
            start = perf_counter()
            try:
                result = request.context.tool_registry.call(request.tool_name, {"query": query})
                elapsed_ms = (perf_counter() - start) * 1000
                search_results.append({"question": question, "result": result})
                if request.context.trace_recorder is not None:
                    request.context.trace_recorder.record(
                        Event(
                            event_type="tool_call",
                            agent=request.agent_name,
                            tool_name=request.tool_name,
                            tool_args={"query": query},
                            tool_result=str(result),
                            latency_ms=elapsed_ms,
                            success=True,
                            metadata=_tool_event_metadata(result),
                        )
                    )
            except Exception as exc:
                elapsed_ms = (perf_counter() - start) * 1000
                error_text = str(exc)
                search_results.append({"question": question, "error": error_text})
                tool_errors.append(error_text)
                if request.context.trace_recorder is not None:
                    request.context.trace_recorder.record(
                        Event(
                            event_type="tool_call",
                            agent=request.agent_name,
                            tool_name=request.tool_name,
                            tool_args={"query": query},
                            tool_result=None,
                            latency_ms=elapsed_ms,
                            success=False,
                            error=error_text,
                            metadata={"provider_errors": _provider_errors_from_text(error_text)},
                        )
                    )

        return SearchStrategyOutput(search_results=search_results, tool_errors=tool_errors)


class SearchAgent(Agent):
    def __init__(
        self,
        model: BaseModel | None = None,
        tool_name: str = "web_search",
        fail_on_tool_error: bool = False,
        strategy: SearchStrategy | None = None,
    ) -> None:
        super().__init__(
            name="searcher",
            role="searcher",
            system_prompt="Search supporting material for each planned question.",
            model=model,
        )
        self.tool_name = tool_name
        self.fail_on_tool_error = fail_on_tool_error
        self.strategy = strategy or DefaultSearchStrategy()

    @property
    def required_services(self) -> set[ServiceName]:
        return {"blackboard", "tool_registry"}

    def run(self, message: Message, context: RuntimeContext) -> Message:
        if context.blackboard is None:
            raise RuntimeError("blackboard is required for SearchAgent")
        if context.tool_registry is None:
            raise RuntimeError("tool_registry is required for SearchAgent")

        plan = read_plan(context.blackboard)
        topic = message.metadata.get("topic", "") if isinstance(message.metadata, dict) else ""
        if not isinstance(topic, str):
            topic = ""

        strategy_output = self.strategy.collect(
            SearchStrategyInput(
                topic=topic,
                plan=plan,
                tool_name=self.tool_name,
                context=context,
                agent_name=self.name,
            )
        )

        write_search_results(
            context.blackboard,
            search_results=strategy_output.search_results,
            author=self.name,
        )
        if strategy_output.tool_errors and self.fail_on_tool_error:
            raise RuntimeError(strategy_output.tool_errors[0])

        return Message(
            sender=self.name,
            receiver=message.sender,
            content=f"Collected {len(strategy_output.search_results)} search results.",
            type="response",
            metadata={"count": len(strategy_output.search_results)},
        )


def _tool_event_metadata(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}

    metadata: dict[str, Any] = {}
    mode = result.get("mode")
    if isinstance(mode, str):
        metadata["search_mode"] = mode

    metadata["fallback_used"] = result.get("fallback_used") is True

    source_hits = _normalize_counter_dict(result.get("source_hits"))
    if source_hits:
        metadata["source_hits"] = source_hits

    provider_errors = _normalize_counter_dict(result.get("provider_errors"))
    if provider_errors:
        metadata["provider_errors"] = provider_errors

    return metadata


def _normalize_counter_dict(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        name = key.strip().lower()
        if name not in {"duckduckgo", "wikipedia", "tavily"}:
            continue
        normalized[name] = _to_int(value)
    return normalized


def _provider_errors_from_text(text: str) -> dict[str, int]:
    lowered = text.lower()
    counts = {"duckduckgo": 0, "wikipedia": 0, "tavily": 0}
    for provider in counts:
        token = f"{provider}_error:"
        counts[provider] = lowered.count(token)
    return {provider: value for provider, value in counts.items() if value > 0}


def _to_int(raw: Any) -> int:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return 0
        try:
            return int(float(text))
        except ValueError:
            return 0
    return 0
