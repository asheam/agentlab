from agentlab.core.agent import Agent
from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message
from agentlab.core.runtime import AgentRuntime
from agentlab.tracing.recorder import TraceRecorder


class EchoAgent(Agent):
    def run(self, message: Message, context: RuntimeContext) -> Message:
        return Message(
            sender=self.name,
            receiver=message.sender,
            content=f"echo: {message.content}",
            type="response",
        )


class BrokenAgent(Agent):
    def run(self, message: Message, context: RuntimeContext) -> Message:
        raise RuntimeError("planned failure")


def test_runtime_routes_message_to_registered_agent() -> None:
    recorder = TraceRecorder()
    runtime = AgentRuntime(trace_recorder=recorder)
    runtime.register_agent(EchoAgent(name="planner", role="planner", system_prompt="x"))

    response = runtime.send(
        Message(sender="user", receiver="planner", content="hello", type="task")
    )

    assert response.type == "response"
    assert response.sender == "planner"
    assert response.receiver == "user"
    assert response.content == "echo: hello"


def test_runtime_returns_error_message_for_missing_agent() -> None:
    runtime = AgentRuntime()

    response = runtime.send(
        Message(sender="user", receiver="missing", content="hello", type="task")
    )

    assert response.type == "error"
    assert "missing" in response.content


def test_runtime_records_trace_events() -> None:
    recorder = TraceRecorder()
    runtime = AgentRuntime(trace_recorder=recorder)
    runtime.register_agent(EchoAgent(name="planner", role="planner", system_prompt="x"))

    runtime.send(Message(sender="user", receiver="planner", content="hello", type="task"))

    payload = recorder.to_dict()
    assert payload["count"] == 1
    assert payload["events"][0]["success"] is True
    assert payload["events"][0]["agent"] == "planner"


def test_runtime_returns_error_when_agent_raises_exception() -> None:
    recorder = TraceRecorder()
    runtime = AgentRuntime(trace_recorder=recorder)
    runtime.register_agent(BrokenAgent(name="broken", role="worker", system_prompt="x"))

    response = runtime.send(
        Message(sender="user", receiver="broken", content="hello", type="task")
    )

    assert response.type == "error"
    assert "planned failure" in response.content

    payload = recorder.to_dict()
    assert payload["count"] == 1
    assert payload["events"][0]["success"] is False
    assert payload["events"][0]["agent"] == "broken"
