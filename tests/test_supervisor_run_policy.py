from __future__ import annotations

import agentlab.multi_agent.supervisor as supervisor_module
from time import sleep

import pytest

from agentlab.core.agent import Agent
from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message
from agentlab.core.runtime import AgentRuntime
from agentlab.multi_agent.scheduler import FixedOrderScheduler
from agentlab.multi_agent.supervisor import RunPolicy, Supervisor
from agentlab.tracing.recorder import TraceRecorder
from agentlab.workspace.artifacts import ArtifactStore
from agentlab.workspace.blackboard import Blackboard


class _FlakyPlanner(Agent):
    def __init__(self) -> None:
        super().__init__(name="planner", role="planner", system_prompt="x")
        self.calls = 0

    def run(self, message: Message, context: RuntimeContext) -> Message:
        del context
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("planned transient failure")
        return Message(
            sender=self.name,
            receiver=message.sender,
            content="planner recovered",
            type="response",
        )


class _AlwaysFailPlanner(Agent):
    def __init__(self) -> None:
        super().__init__(name="planner", role="planner", system_prompt="x")
        self.calls = 0

    def run(self, message: Message, context: RuntimeContext) -> Message:
        del message, context
        self.calls += 1
        raise RuntimeError("planned failure")


class _SlowPlanner(Agent):
    def __init__(self) -> None:
        super().__init__(name="planner", role="planner", system_prompt="x")
        self.calls = 0

    def run(self, message: Message, context: RuntimeContext) -> Message:
        del context
        self.calls += 1
        sleep(0.03)
        return Message(
            sender=self.name,
            receiver=message.sender,
            content="slow response",
            type="response",
        )


class _FailPlanner(Agent):
    def __init__(self) -> None:
        super().__init__(name="planner", role="planner", system_prompt="x")

    def run(self, message: Message, context: RuntimeContext) -> Message:
        del message, context
        raise RuntimeError("planner hard fail")


class _WriterAgent(Agent):
    def __init__(self) -> None:
        super().__init__(name="writer", role="writer", system_prompt="x")

    def run(self, message: Message, context: RuntimeContext) -> Message:
        del context
        return Message(
            sender=self.name,
            receiver=message.sender,
            content="writer completed",
            type="response",
        )


def _build_runtime(tmp_path) -> AgentRuntime:
    return AgentRuntime(
        blackboard=Blackboard(),
        trace_recorder=TraceRecorder(),
        artifacts=ArtifactStore(base_dir=tmp_path),
    )


def test_run_policy_rejects_negative_retry_backoff() -> None:
    with pytest.raises(ValueError, match="retry_backoff_s must be >= 0"):
        RunPolicy(retry_backoff_s=-0.1)


def test_supervisor_retries_and_recovers(tmp_path) -> None:
    runtime = _build_runtime(tmp_path)
    planner = _FlakyPlanner()
    runtime.register_agent(planner)

    supervisor = Supervisor(
        runtime=runtime,
        scheduler=FixedOrderScheduler(order=["planner"]),
        run_policy=RunPolicy(max_retries=1),
    )
    outputs = supervisor.run("topic")

    assert planner.calls == 2
    assert outputs.report_path.exists()
    assert "planner recovered" in outputs.report_path.read_text(encoding="utf-8")
    payload = runtime.trace_recorder.to_dict()
    assert any(
        event.get("metadata", {}).get("reason") == "retry_recovered"
        for event in payload["events"]
    )
    summary = runtime.trace_recorder.to_summary_dict()
    assert summary["retry_stats"]["total_retries"] == 1
    assert summary["retry_stats"]["error_retries"] == 1


def test_supervisor_retries_exhausted_raises(tmp_path) -> None:
    runtime = _build_runtime(tmp_path)
    planner = _AlwaysFailPlanner()
    runtime.register_agent(planner)

    supervisor = Supervisor(
        runtime=runtime,
        scheduler=FixedOrderScheduler(order=["planner"]),
        run_policy=RunPolicy(max_retries=1),
    )
    with pytest.raises(RuntimeError, match="planned failure"):
        supervisor.run("topic")

    assert planner.calls == 2
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "trace.json").exists()
    assert (tmp_path / "workspace.json").exists()
    assert (tmp_path / "run_summary.json").exists()


def test_supervisor_continue_on_error_keeps_pipeline_running(tmp_path) -> None:
    runtime = _build_runtime(tmp_path)
    runtime.register_agent(_FailPlanner())
    runtime.register_agent(_WriterAgent())

    supervisor = Supervisor(
        runtime=runtime,
        scheduler=FixedOrderScheduler(order=["planner", "writer"]),
        run_policy=RunPolicy(continue_on_error=True),
    )
    outputs = supervisor.run("topic")

    assert outputs.report_path.exists()
    assert "writer completed" in outputs.report_path.read_text(encoding="utf-8")
    payload = runtime.trace_recorder.to_dict()
    assert any(
        event.get("metadata", {}).get("reason") == "continue_on_error"
        for event in payload["events"]
    )


def test_supervisor_soft_timeout_triggers_retry_and_failure(tmp_path) -> None:
    runtime = _build_runtime(tmp_path)
    planner = _SlowPlanner()
    runtime.register_agent(planner)

    supervisor = Supervisor(
        runtime=runtime,
        scheduler=FixedOrderScheduler(order=["planner"]),
        run_policy=RunPolicy(max_retries=1, agent_timeout_s=0.003),
    )
    with pytest.raises(RuntimeError, match="timed out"):
        supervisor.run("topic")

    assert planner.calls == 2
    payload = runtime.trace_recorder.to_dict()
    timeout_events = [
        event
        for event in payload["events"]
        if event.get("metadata", {}).get("reason") == "timeout"
    ]
    assert len(timeout_events) >= 1
    summary = runtime.trace_recorder.to_summary_dict()
    assert summary["retry_stats"]["timeout_retries"] == 1


def test_supervisor_retry_on_timeout_only_disables_runtime_error_retries(tmp_path) -> None:
    runtime = _build_runtime(tmp_path)
    planner = _AlwaysFailPlanner()
    runtime.register_agent(planner)

    supervisor = Supervisor(
        runtime=runtime,
        scheduler=FixedOrderScheduler(order=["planner"]),
        run_policy=RunPolicy(max_retries=2, retry_on_timeout_only=True),
    )
    with pytest.raises(RuntimeError, match="planned failure"):
        supervisor.run("topic")

    assert planner.calls == 1
    summary = runtime.trace_recorder.to_summary_dict()
    assert summary["retry_stats"]["total_retries"] == 0
    assert summary["retry_stats"]["error_retries"] == 0


def test_supervisor_applies_retry_backoff_between_attempts(monkeypatch, tmp_path) -> None:
    runtime = _build_runtime(tmp_path)
    planner = _FlakyPlanner()
    runtime.register_agent(planner)

    calls: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        calls.append(seconds)

    monkeypatch.setattr(supervisor_module, "sleep", _fake_sleep)

    supervisor = Supervisor(
        runtime=runtime,
        scheduler=FixedOrderScheduler(order=["planner"]),
        run_policy=RunPolicy(max_retries=1, retry_backoff_s=0.05),
    )
    supervisor.run("topic")

    assert planner.calls == 2
    assert calls == [0.05]


def test_timeout_trace_events_are_distinguishable_by_emitter(tmp_path) -> None:
    runtime = _build_runtime(tmp_path)
    runtime.register_agent(_SlowPlanner())

    supervisor = Supervisor(
        runtime=runtime,
        scheduler=FixedOrderScheduler(order=["planner"]),
        run_policy=RunPolicy(max_retries=0, agent_timeout_s=0.003),
    )
    with pytest.raises(RuntimeError, match="timed out"):
        supervisor.run("topic")

    payload = runtime.trace_recorder.to_dict()
    runtime_success_events = [
        event
        for event in payload["events"]
        if event.get("agent") == "planner"
        and event.get("success") is True
        and event.get("metadata", {}).get("emitter") is None
    ]
    supervisor_timeout_events = [
        event
        for event in payload["events"]
        if event.get("agent") == "planner"
        and event.get("metadata", {}).get("emitter") == "supervisor"
        and event.get("metadata", {}).get("reason") == "timeout"
    ]
    assert len(runtime_success_events) >= 1
    assert len(supervisor_timeout_events) >= 1
