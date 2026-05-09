from __future__ import annotations

from time import perf_counter

from agentlab.core.agent import Agent
from agentlab.core.context import RuntimeContext
from agentlab.core.event import Event
from agentlab.core.message import Message
from agentlab.tools.registry import ToolRegistry
from agentlab.tracing.recorder import TraceRecorder
from agentlab.workspace.artifacts import ArtifactStore
from agentlab.workspace.blackboard import Blackboard


class AgentRuntime:
    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        blackboard: Blackboard | None = None,
        trace_recorder: TraceRecorder | None = None,
        artifacts: ArtifactStore | None = None,
    ) -> None:
        self._agents: dict[str, Agent] = {}
        self.tool_registry = tool_registry or ToolRegistry()
        self.blackboard = blackboard or Blackboard()
        self.trace_recorder = trace_recorder or TraceRecorder()
        self.artifacts = artifacts or ArtifactStore()

    def register_agent(self, agent: Agent) -> None:
        self._agents[agent.name] = agent

    def send(self, message: Message) -> Message:
        target_agent = self._agents.get(message.receiver)
        if target_agent is None:
            return self._missing_agent_message(message)

        context = RuntimeContext(
            runtime=self,
            tool_registry=self.tool_registry,
            blackboard=self.blackboard,
            trace_recorder=self.trace_recorder,
            artifacts=self.artifacts,
        )
        missing_services = [
            service for service in target_agent.required_services if getattr(context, service) is None
        ]
        if missing_services:
            error_text = (
                f"Agent '{target_agent.name}' missing required services: "
                + ", ".join(sorted(missing_services))
            )
            self.trace_recorder.record(
                Event(
                    event_type="agent_call",
                    agent=target_agent.name,
                    input=message.content,
                    output=error_text,
                    latency_ms=0.0,
                    success=False,
                    error=error_text,
                    metadata={
                        "sender": message.sender,
                        "message_type": message.type,
                        "missing_services": sorted(missing_services),
                    },
                )
            )
            return Message(
                sender="runtime",
                receiver=message.sender,
                content=error_text,
                type="error",
                metadata={"failed_agent": target_agent.name, "missing_services": missing_services},
            )

        start_time = perf_counter()
        try:
            response = target_agent.run(message, context)
            latency_ms = (perf_counter() - start_time) * 1000
            self.trace_recorder.record(
                Event(
                    event_type="agent_call",
                    agent=target_agent.name,
                    input=message.content,
                    output=response.content,
                    latency_ms=latency_ms,
                    success=True,
                    metadata={"sender": message.sender, "message_type": message.type},
                )
            )
            return response
        except Exception as exc:
            latency_ms = (perf_counter() - start_time) * 1000
            error_text = f"Agent '{target_agent.name}' failed: {exc}"
            self.trace_recorder.record(
                Event(
                    event_type="agent_call",
                    agent=target_agent.name,
                    input=message.content,
                    output=error_text,
                    latency_ms=latency_ms,
                    success=False,
                    error=str(exc),
                    metadata={"sender": message.sender, "message_type": message.type},
                )
            )
            return Message(
                sender="runtime",
                receiver=message.sender,
                content=error_text,
                type="error",
                metadata={"failed_agent": target_agent.name},
            )

    def _missing_agent_message(self, message: Message) -> Message:
        error_message = Message(
            sender="runtime",
            receiver=message.sender,
            content=f"Agent '{message.receiver}' is not registered",
            type="error",
            metadata={"missing_agent": message.receiver},
        )
        self.trace_recorder.record(
            Event(
                event_type="agent_call",
                agent=message.receiver,
                input=message.content,
                output=error_message.content,
                latency_ms=0.0,
                success=False,
                error=error_message.content,
                metadata={"sender": message.sender, "message_type": message.type},
            )
        )
        return error_message
