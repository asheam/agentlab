from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Literal

from dotenv import load_dotenv

from agentlab.agents import (
    CriticAgent,
    PlannerAgent,
    ReaderAgent,
    SearchAgent,
    WriterAgent,
)
from agentlab.agents.planner_agent import PlannerStrategy
from agentlab.agents.writer_agent import WriterStrategy
from agentlab.core.agent import Agent
from agentlab.core.event import Event
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
class RunPolicy:
    max_retries: int = 0
    agent_timeout_s: float | None = None
    continue_on_error: bool = False

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.agent_timeout_s is not None and self.agent_timeout_s <= 0:
            raise ValueError("agent_timeout_s must be > 0 when provided")


@dataclass(frozen=True)
class SupervisorConfig:
    output_dir: str | Path = "outputs"
    use_openai_model: bool = False
    search_mode: str = "mock"
    allow_search_fallback: bool = True
    search_providers: list[str] | tuple[str, ...] | None = None
    critic_mode: Literal["auto", "rule", "llm"] = "auto"
    planner_strategy: PlannerStrategy | None = None
    writer_strategy: WriterStrategy | None = None
    run_policy: RunPolicy = field(default_factory=RunPolicy)


@dataclass(frozen=True)
class SupervisorOutput:
    report_path: Path
    trace_path: Path
    workspace_path: Path
    summary_path: Path

    def items(self) -> list[tuple[str, Path]]:
        return [
            ("report_path", self.report_path),
            ("trace_path", self.trace_path),
            ("workspace_path", self.workspace_path),
            ("summary_path", self.summary_path),
        ]


class Supervisor:
    def __init__(
        self,
        runtime: AgentRuntime,
        scheduler: Scheduler | None = None,
        run_policy: RunPolicy | None = None,
    ) -> None:
        self.runtime = runtime
        self.scheduler = scheduler or FixedOrderScheduler()
        self.run_policy = run_policy or RunPolicy()

    def run(self, topic: str) -> SupervisorOutput:
        sender = "supervisor"
        latest_response: Message | None = None
        run_error: RuntimeError | None = None

        for agent_name in self.scheduler.get_order():
            latest_response = self._execute_agent_with_policy(
                agent_name=agent_name,
                sender=sender,
                topic=topic,
            )
            if latest_response.type == "error":
                if not self.run_policy.continue_on_error:
                    run_error = RuntimeError(latest_response.content)
                    break
            else:
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
        summary_path = self.runtime.trace_recorder.export_summary_json(
            self.runtime.artifacts.base_dir / "run_summary.json"
        )

        outputs = SupervisorOutput(
            report_path=report_path,
            trace_path=trace_path,
            workspace_path=workspace_path,
            summary_path=summary_path,
        )
        if run_error is not None:
            raise run_error
        return outputs

    def _execute_agent_with_policy(self, agent_name: AgentName, sender: str, topic: str) -> Message:
        max_attempts = self.run_policy.max_retries + 1
        last_error: Message | None = None
        for attempt in range(1, max_attempts + 1):
            request = Message(
                sender=sender,
                receiver=agent_name,
                content=_instruction_for(agent_name, topic),
                type="task",
                metadata={"topic": topic, "attempt": attempt, "max_attempts": max_attempts},
            )

            started = perf_counter()
            response = self.runtime.send(request)
            elapsed_s = perf_counter() - started

            timeout_s = self.run_policy.agent_timeout_s
            if timeout_s is not None and elapsed_s > timeout_s:
                # Soft timeout: runtime call already returned, so partial side effects may exist.
                # Current retry behavior relies on overwrite semantics in workspace writes.
                timeout_error = (
                    f"Agent '{agent_name}' timed out: elapsed={elapsed_s:.3f}s > "
                    f"timeout={timeout_s:.3f}s (attempt {attempt}/{max_attempts})"
                )
                self._record_supervisor_event(
                    agent_name=agent_name,
                    success=False,
                    output=timeout_error,
                    error=timeout_error,
                    metadata={
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "reason": "timeout",
                        "elapsed_s": round(elapsed_s, 6),
                        "timeout_s": timeout_s,
                    },
                )
                last_error = Message(
                    sender="supervisor",
                    receiver=sender,
                    content=timeout_error,
                    type="error",
                    metadata={
                        "failed_agent": agent_name,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "elapsed_s": elapsed_s,
                        "timeout_s": timeout_s,
                    },
                )
            elif response.type == "error":
                self._record_supervisor_event(
                    agent_name=agent_name,
                    success=False,
                    output=response.content,
                    error=response.content,
                    metadata={
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "reason": "runtime_error",
                    },
                )
                last_error = response
            else:
                if attempt > 1:
                    self._record_supervisor_event(
                        agent_name=agent_name,
                        success=True,
                        output=response.content,
                        error=None,
                        metadata={
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "reason": "retry_recovered",
                        },
                    )
                return response

            if attempt < max_attempts:
                continue

        assert last_error is not None
        if self.run_policy.continue_on_error:
            self._record_supervisor_event(
                agent_name=agent_name,
                success=False,
                output=last_error.content,
                error=last_error.content,
                metadata={"reason": "continue_on_error"},
            )
        return last_error

    def _record_supervisor_event(
        self,
        agent_name: AgentName,
        success: bool,
        output: str,
        error: str | None,
        metadata: dict[str, object],
    ) -> None:
        self.runtime.trace_recorder.record(
            Event(
                event_type="agent_call",
                agent=agent_name,
                output=output,
                success=success,
                error=error,
                metadata={"emitter": "supervisor", **metadata},
            )
        )


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
        PlannerAgent(model=model, strategy=config.planner_strategy),
        SearchAgent(model=model, fail_on_tool_error=strict_real_search),
        ReaderAgent(model=model),
        CriticAgent(model=model, critic_mode=config.critic_mode),
        WriterAgent(model=model, strategy=config.writer_strategy),
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
    return Supervisor(
        runtime=runtime,
        scheduler=FixedOrderScheduler(),
        run_policy=effective_config.run_policy,
    )


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
