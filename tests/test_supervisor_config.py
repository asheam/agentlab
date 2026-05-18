from __future__ import annotations

import pytest

from agentlab.agents import CriticAgent, PlannerAgent, ReaderAgent, SearchAgent, WriterAgent
from agentlab.agents.critic_agent import CriticStrategyInput
from agentlab.agents.reader_agent import ReaderStrategyInput
from agentlab.agents.search_agent import SearchStrategyInput, SearchStrategyOutput
from agentlab.agents.writer_agent import WriterStrategyInput
from agentlab.models.base import BaseModel
from agentlab.multi_agent.supervisor import (
    SupervisorConfig,
    build_default_agents,
    build_default_supervisor,
    build_default_tools,
)
from agentlab.workspace.research_workspace import CritiquePayload, NotesPayload


def test_build_default_supervisor_supports_config_object(tmp_path) -> None:
    config = SupervisorConfig(output_dir=tmp_path)
    supervisor = build_default_supervisor(config=config)

    outputs = supervisor.run("Research LangGraph AutoGen CrewAI differences")

    assert outputs.report_path.exists()
    assert outputs.trace_path.exists()
    assert outputs.workspace_path.exists()
    assert outputs.summary_path.exists()


def test_build_default_supervisor_legacy_kwargs_emit_deprecation_warning(tmp_path) -> None:
    with pytest.warns(DeprecationWarning, match="legacy kwargs are deprecated"):
        build_default_supervisor(output_dir=tmp_path)


def test_build_default_supervisor_rejects_mixed_config_and_legacy_kwargs(tmp_path) -> None:
    config = SupervisorConfig(output_dir=tmp_path / "config_out")
    with pytest.raises(ValueError, match="both 'config' and legacy kwargs"):
        build_default_supervisor(config=config, output_dir=tmp_path / "legacy_out")


def test_build_default_tools_registers_default_set() -> None:
    config = SupervisorConfig(search_mode="mock")
    registry = build_default_tools(config)

    assert registry.list_tools() == ["calculator", "web_search"]


def test_build_default_agents_returns_expected_sequence() -> None:
    config = SupervisorConfig(search_mode="real", allow_search_fallback=False)
    agents = build_default_agents(model=None, config=config)

    assert [agent.name for agent in agents] == ["planner", "searcher", "reader", "critic", "writer"]
    assert isinstance(agents[0], PlannerAgent)
    assert isinstance(agents[1], SearchAgent)
    assert isinstance(agents[2], ReaderAgent)
    assert isinstance(agents[3], CriticAgent)
    assert isinstance(agents[4], WriterAgent)
    assert agents[1].fail_on_tool_error is True


class _StaticPlannerStrategy:
    def build_plan(self, topic: str, model: BaseModel | None) -> list[str]:
        del topic, model
        return ["custom plan question"]


class _StaticWriterStrategy:
    def build_report(self, request: WriterStrategyInput) -> str:
        del request
        return "# Custom Report\n"


class _StaticSearchStrategy:
    def collect(self, request: SearchStrategyInput) -> SearchStrategyOutput:
        del request
        return SearchStrategyOutput(search_results=[], tool_errors=[])


class _StaticReaderStrategy:
    def build_notes(self, request: ReaderStrategyInput) -> NotesPayload:
        del request
        return NotesPayload(key_points=[], references=[], summary="x", structured={})


class _StaticCriticStrategy:
    def build_critique(self, request: CriticStrategyInput) -> CritiquePayload:
        del request
        return CritiquePayload(verdict="acceptable", assessment_mode="custom")


def test_build_default_agents_injects_custom_strategies() -> None:
    planner_strategy = _StaticPlannerStrategy()
    search_strategy = _StaticSearchStrategy()
    reader_strategy = _StaticReaderStrategy()
    critic_strategy = _StaticCriticStrategy()
    writer_strategy = _StaticWriterStrategy()
    config = SupervisorConfig(
        planner_strategy=planner_strategy,
        search_strategy=search_strategy,
        reader_strategy=reader_strategy,
        critic_strategy=critic_strategy,
        writer_strategy=writer_strategy,
    )
    agents = build_default_agents(model=None, config=config)

    assert isinstance(agents[0], PlannerAgent)
    assert agents[0].strategy is planner_strategy
    assert isinstance(agents[1], SearchAgent)
    assert agents[1].strategy is search_strategy
    assert isinstance(agents[2], ReaderAgent)
    assert agents[2].strategy is reader_strategy
    assert isinstance(agents[3], CriticAgent)
    assert agents[3].strategy is critic_strategy
    assert isinstance(agents[4], WriterAgent)
    assert agents[4].strategy is writer_strategy
