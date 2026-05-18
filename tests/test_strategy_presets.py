from __future__ import annotations

from agentlab.multi_agent.supervisor import SupervisorConfig
from agentlab.strategy_presets import (
    ConciseCriticStrategy,
    ConcisePlannerStrategy,
    ConciseReaderStrategy,
    ConciseSearchStrategy,
    ConciseWriterStrategy,
    apply_strategy_preset,
    iter_strategy_presets,
)


def test_apply_default_strategy_preset_keeps_config() -> None:
    config = SupervisorConfig(search_mode="mock", critic_mode="auto")
    applied = apply_strategy_preset(config, "default")

    assert applied is config
    assert applied.planner_strategy is None
    assert applied.search_strategy is None
    assert applied.reader_strategy is None
    assert applied.critic_strategy is None
    assert applied.writer_strategy is None


def test_apply_concise_strategy_preset_injects_all_strategies() -> None:
    config = SupervisorConfig(search_mode="real", critic_mode="llm")
    applied = apply_strategy_preset(config, "concise")

    assert isinstance(applied.planner_strategy, ConcisePlannerStrategy)
    assert isinstance(applied.search_strategy, ConciseSearchStrategy)
    assert isinstance(applied.reader_strategy, ConciseReaderStrategy)
    assert isinstance(applied.critic_strategy, ConciseCriticStrategy)
    assert isinstance(applied.writer_strategy, ConciseWriterStrategy)
    assert applied.search_mode == "real"
    assert applied.critic_mode == "llm"


def test_iter_strategy_presets_contains_default_and_concise() -> None:
    items = iter_strategy_presets()
    names = [name for name, _ in items]

    assert names == ["default", "concise"]
    assert "built-in default" in items[0][1]
    assert "compact outputs" in items[1][1]
