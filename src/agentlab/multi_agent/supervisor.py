from __future__ import annotations

from pathlib import Path

from agentlab.agents import (
    CriticAgent,
    PlannerAgent,
    ReaderAgent,
    SearchAgent,
    WriterAgent,
)
from agentlab.core.message import Message
from agentlab.core.runtime import AgentRuntime
from agentlab.multi_agent.scheduler import FixedOrderScheduler
from agentlab.multi_agent.team import AgentTeam
from agentlab.tools.calculator import CalculatorTool
from agentlab.tools.registry import ToolRegistry
from agentlab.tools.web_search import WebSearchTool
from agentlab.tracing.recorder import TraceRecorder
from agentlab.workspace.artifacts import ArtifactStore
from agentlab.workspace.blackboard import Blackboard


class Supervisor:
    def __init__(self, runtime: AgentRuntime, scheduler: FixedOrderScheduler | None = None) -> None:
        self.runtime = runtime
        self.scheduler = scheduler or FixedOrderScheduler()

    def run(self, topic: str) -> dict[str, Path]:
        sender = "supervisor"
        latest_response: Message | None = None

        for agent_name in self.scheduler.get_order():
            request = Message(
                sender=sender,
                receiver=agent_name,
                content=_instruction_for(agent_name, topic),
                type="task",
                metadata={"topic": topic},
            )
            latest_response = self.runtime.send(request)
            if latest_response.type == "error":
                raise RuntimeError(latest_response.content)
            sender = agent_name

        report = self.runtime.blackboard.read("report", "")
        if not isinstance(report, str):
            report = str(report)
        if not report and latest_response is not None:
            report = latest_response.content

        report_path = self.runtime.artifacts.save_text("report.md", report)
        trace_path = self.runtime.trace_recorder.export_json(
            self.runtime.artifacts.base_dir / "trace.json"
        )
        workspace_path = self.runtime.blackboard.export_json(
            self.runtime.artifacts.base_dir / "workspace.json"
        )

        return {
            "report_path": report_path,
            "trace_path": trace_path,
            "workspace_path": workspace_path,
        }


def build_default_supervisor(output_dir: str | Path = "outputs") -> Supervisor:
    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(WebSearchTool())

    runtime = AgentRuntime(
        tool_registry=tool_registry,
        blackboard=Blackboard(),
        trace_recorder=TraceRecorder(),
        artifacts=ArtifactStore(base_dir=output_dir),
    )

    team = AgentTeam(runtime)
    team.add_many(
        [
            PlannerAgent(),
            SearchAgent(),
            ReaderAgent(),
            CriticAgent(),
            WriterAgent(),
        ]
    )

    return Supervisor(runtime=runtime, scheduler=FixedOrderScheduler())


def _instruction_for(agent_name: str, topic: str) -> str:
    if agent_name == "planner":
        return topic
    if agent_name == "searcher":
        return "请基于 plan 检索信息并写入 search_results。"
    if agent_name == "reader":
        return "请阅读 search_results 并整理 notes。"
    if agent_name == "critic":
        return "请审查 notes 的完整性并写入 critique。"
    if agent_name == "writer":
        return topic
    return topic
