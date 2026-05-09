from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from agentlab.agents import (
    CriticAgent,
    PlannerAgent,
    ReaderAgent,
    SearchAgent,
    WriterAgent,
)
from agentlab.core.message import Message
from agentlab.core.runtime import AgentRuntime
from agentlab.models.base import BaseModel
from agentlab.models.openai_compatible import OpenAICompatibleModel
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
        run_error: RuntimeError | None = None

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
                run_error = RuntimeError(latest_response.content)
                break
            sender = agent_name

        report = self.runtime.blackboard.read("report", "")
        if not isinstance(report, str):
            report = str(report)
        if not report and latest_response is not None:
            report = latest_response.content
        if run_error is not None and not report.strip():
            report = _build_error_report(topic, str(run_error))

        report_path = self.runtime.artifacts.save_text("report.md", report)
        trace_path = self.runtime.trace_recorder.export_json(
            self.runtime.artifacts.base_dir / "trace.json"
        )
        workspace_path = self.runtime.blackboard.export_json(
            self.runtime.artifacts.base_dir / "workspace.json"
        )

        outputs = {
            "report_path": report_path,
            "trace_path": trace_path,
            "workspace_path": workspace_path,
        }
        if run_error is not None:
            raise run_error
        return outputs


def _build_error_report(topic: str, error: str) -> str:
    return (
        "# Deep Research Report\n\n"
        f"## Topic\n{topic}\n\n"
        "## Runtime Error\n"
        f"- {error}\n"
    )


def build_default_supervisor(
    output_dir: str | Path = "outputs",
    use_openai_model: bool = False,
    search_mode: str = "mock",
    allow_search_fallback: bool = True,
    search_providers: list[str] | tuple[str, ...] | None = None,
) -> Supervisor:
    # Ensure .env variables (e.g., OPENAI_API_KEY / TAVILY_API_KEY) are available.
    load_dotenv()

    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(
        WebSearchTool(
            mode=search_mode,
            allow_fallback=allow_search_fallback,
            real_providers=search_providers,
        )
    )

    runtime = AgentRuntime(
        tool_registry=tool_registry,
        blackboard=Blackboard(),
        trace_recorder=TraceRecorder(),
        artifacts=ArtifactStore(base_dir=output_dir),
    )

    model: BaseModel | None = OpenAICompatibleModel() if use_openai_model else None

    team = AgentTeam(runtime)
    strict_real_search = search_mode == "real" and not allow_search_fallback
    team.add_many(
        [
            PlannerAgent(model=model),
            SearchAgent(model=model, fail_on_tool_error=strict_real_search),
            ReaderAgent(model=model),
            CriticAgent(model=model),
            WriterAgent(model=model),
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
