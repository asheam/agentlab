from __future__ import annotations

import pytest

from agentlab.agents import CriticAgent, PlannerAgent, ReaderAgent, SearchAgent, WriterAgent
from agentlab.multi_agent.supervisor import (
    SupervisorConfig,
    build_default_agents,
    build_default_supervisor,
    build_default_tools,
)


def test_build_default_supervisor_supports_config_object(tmp_path) -> None:
    config = SupervisorConfig(output_dir=tmp_path)
    supervisor = build_default_supervisor(config=config)

    outputs = supervisor.run("研究 LangGraph、AutoGen、CrewAI 的区别")

    assert outputs.report_path.exists()
    assert outputs.trace_path.exists()
    assert outputs.workspace_path.exists()


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
