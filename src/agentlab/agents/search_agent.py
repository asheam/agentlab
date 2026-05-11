from __future__ import annotations

from time import perf_counter

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


class SearchAgent(Agent):
    def __init__(
        self,
        model: BaseModel | None = None,
        tool_name: str = "web_search",
        fail_on_tool_error: bool = False,
    ) -> None:
        super().__init__(
            name="searcher",
            role="searcher",
            system_prompt="Search supporting material for each planned question.",
            model=model,
        )
        self.tool_name = tool_name
        self.fail_on_tool_error = fail_on_tool_error

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

        search_results: list[SearchResultItem] = []
        tool_errors: list[str] = []
        for question in plan:
            query = f"{topic} {question}".strip() if topic else question
            start = perf_counter()
            try:
                result = context.tool_registry.call(self.tool_name, {"query": query})
                elapsed_ms = (perf_counter() - start) * 1000
                search_results.append({"question": question, "result": result})
                if context.trace_recorder is not None:
                    context.trace_recorder.record(
                        Event(
                            event_type="tool_call",
                            agent=self.name,
                            tool_name=self.tool_name,
                            tool_args={"query": query},
                            tool_result=str(result),
                            latency_ms=elapsed_ms,
                            success=True,
                        )
                    )
            except Exception as exc:
                elapsed_ms = (perf_counter() - start) * 1000
                error_text = str(exc)
                search_results.append({"question": question, "error": error_text})
                tool_errors.append(error_text)
                if context.trace_recorder is not None:
                    context.trace_recorder.record(
                        Event(
                            event_type="tool_call",
                            agent=self.name,
                            tool_name=self.tool_name,
                            tool_args={"query": query},
                            tool_result=None,
                            latency_ms=elapsed_ms,
                            success=False,
                            error=error_text,
                        )
                    )

        write_search_results(context.blackboard, search_results=search_results, author=self.name)
        if tool_errors and self.fail_on_tool_error:
            raise RuntimeError(tool_errors[0])

        return Message(
            sender=self.name,
            receiver=message.sender,
            content=f"Collected {len(search_results)} search results.",
            type="response",
            metadata={"count": len(search_results)},
        )
