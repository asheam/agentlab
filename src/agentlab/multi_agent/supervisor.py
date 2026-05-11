from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import warnings

from dotenv import load_dotenv

from agentlab.agents import (
    CriticAgent,
    PlannerAgent,
    ReaderAgent,
    SearchAgent,
    WriterAgent,
)
from agentlab.core.agent import Agent
from agentlab.core.message import Message
from agentlab.core.runtime import AgentRuntime
from agentlab.models.base import BaseModel
from agentlab.models.openai_compatible import OpenAICompatibleModel
from agentlab.multi_agent.scheduler import AgentName, FixedOrderScheduler, Scheduler
from agentlab.multi_agent.team import AgentTeam
from agentlab.tools.calculator import CalculatorTool
from agentlab.tools.registry import ToolRegistry
from agentlab.tools.web_search import WebSearchTool
from agentlab.tracing.recorder import TraceRecorder
from agentlab.workspace.artifacts import ArtifactStore
from agentlab.workspace.blackboard import Blackboard
from agentlab.workspace.research_workspace import read_report


@dataclass(frozen=True)
class SupervisorConfig:
    output_dir: str | Path = "outputs"
    use_openai_model: bool = False
    search_mode: str = "mock"
    allow_search_fallback: bool = True
    search_providers: list[str] | tuple[str, ...] | None = None
    critic_mode: Literal["auto", "rule", "llm"] = "auto"


@dataclass(frozen=True)
class SupervisorOutput:
    report_path: Path
    trace_path: Path
    workspace_path: Path

    def items(self) -> list[tuple[str, Path]]:
        return [
            ("report_path", self.report_path),
            ("trace_path", self.trace_path),
            ("workspace_path", self.workspace_path),
        ]


class Supervisor:
    def __init__(self, runtime: AgentRuntime, scheduler: Scheduler | None = None) -> None:
        self.runtime = runtime
        self.scheduler = scheduler or FixedOrderScheduler()

    def run(self, topic: str) -> SupervisorOutput:
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

        report = read_report(self.runtime.blackboard)
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

        outputs = SupervisorOutput(
            report_path=report_path,
            trace_path=trace_path,
            workspace_path=workspace_path,
        )
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


def build_default_tools(config: SupervisorConfig) -> ToolRegistry:
    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(
        WebSearchTool(
            mode=config.search_mode,
            allow_fallback=config.allow_search_fallback,
            real_providers=config.search_providers,
        )
    )
    return tool_registry


def build_default_agents(model: BaseModel | None, config: SupervisorConfig) -> list[Agent]:
    strict_real_search = config.search_mode == "real" and not config.allow_search_fallback
    return [
        PlannerAgent(model=model),
        SearchAgent(model=model, fail_on_tool_error=strict_real_search),
        ReaderAgent(model=model),
        CriticAgent(model=model, critic_mode=config.critic_mode),
        WriterAgent(model=model),
    ]


def build_default_supervisor(
    output_dir: str | Path = "outputs",
    use_openai_model: bool = False,
    search_mode: str = "mock",
    allow_search_fallback: bool = True,
    search_providers: list[str] | tuple[str, ...] | None = None,
    critic_mode: Literal["auto", "rule", "llm"] = "auto",
    *,
    config: SupervisorConfig | None = None,
) -> Supervisor:
    # Ensure .env variables (e.g., OPENAI_API_KEY / TAVILY_API_KEY) are available.
    load_dotenv()

    legacy_config = SupervisorConfig(
        output_dir=output_dir,
        use_openai_model=use_openai_model,
        search_mode=search_mode,
        allow_search_fallback=allow_search_fallback,
        search_providers=search_providers,
        critic_mode=critic_mode,
    )

    if config is not None:
        if _legacy_config_changed_from_defaults(legacy_config):
            raise ValueError(
                "build_default_supervisor received both 'config' and legacy kwargs. "
                "Please use only SupervisorConfig."
            )
        effective_config = config
    else:
        effective_config = legacy_config
        if _legacy_config_changed_from_defaults(legacy_config):
            warnings.warn(
                "build_default_supervisor legacy kwargs are deprecated and will be removed "
                "in a future version. Use SupervisorConfig via `config=`.",
                DeprecationWarning,
                stacklevel=2,
            )

    tool_registry = build_default_tools(effective_config)

    runtime = AgentRuntime(
        tool_registry=tool_registry,
        blackboard=Blackboard(),
        trace_recorder=TraceRecorder(),
        artifacts=ArtifactStore(base_dir=effective_config.output_dir),
    )

    model: BaseModel | None = OpenAICompatibleModel() if effective_config.use_openai_model else None

    team = AgentTeam(runtime)
    team.add_many(build_default_agents(model=model, config=effective_config))
    return Supervisor(runtime=runtime, scheduler=FixedOrderScheduler())


def _legacy_config_changed_from_defaults(config: SupervisorConfig) -> bool:
    defaults = SupervisorConfig()
    return (
        Path(config.output_dir) != Path(defaults.output_dir)
        or config.use_openai_model != defaults.use_openai_model
        or config.search_mode != defaults.search_mode
        or config.allow_search_fallback != defaults.allow_search_fallback
        or tuple(config.search_providers or ()) != tuple(defaults.search_providers or ())
        or config.critic_mode != defaults.critic_mode
    )


def _instruction_for(agent_name: AgentName, topic: str) -> str:
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
