from __future__ import annotations

import pytest

from agentlab.cli_config import CLIConfigError, load_cli_config


def test_load_cli_config_parses_basic_yaml(tmp_path) -> None:
    config_path = tmp_path / "agentlab.yaml"
    config_path.write_text(
        "\n".join(
            [
                "topic: research topic",
                "output_dir: out",
                "search_mode: real",
                "search_providers:",
                "  - tavily",
                "  - duckduckgo",
                "critic_mode: llm",
                "strategy_preset: concise",
                "no_search_fallback: true",
                "use_openai: false",
                "max_retries: 2",
                "agent_timeout_s: 9.5",
                "retry_backoff_s: 0.2",
                "retry_on_timeout_only: true",
                "continue_on_error: false",
            ]
        ),
        encoding="utf-8",
    )

    data = load_cli_config(config_path)

    assert data["topic"] == "research topic"
    assert data["output_dir"] == "out"
    assert data["search_mode"] == "real"
    assert data["search_providers"] == ["tavily", "duckduckgo"]
    assert data["critic_mode"] == "llm"
    assert data["strategy_preset"] == "concise"
    assert data["no_search_fallback"] is True
    assert data["use_openai"] is False
    assert data["max_retries"] == 2
    assert data["agent_timeout_s"] == 9.5
    assert data["retry_backoff_s"] == 0.2
    assert data["retry_on_timeout_only"] is True
    assert data["continue_on_error"] is False


def test_load_cli_config_rejects_unknown_keys(tmp_path) -> None:
    config_path = tmp_path / "agentlab.yaml"
    config_path.write_text("unknown_key: value\n", encoding="utf-8")

    with pytest.raises(CLIConfigError, match="Unknown config keys"):
        load_cli_config(config_path)


def test_load_cli_config_rejects_missing_file() -> None:
    with pytest.raises(CLIConfigError, match="Config file not found"):
        load_cli_config("missing-agentlab.yaml")


def test_load_cli_config_rejects_invalid_run_policy_values(tmp_path) -> None:
    config_path = tmp_path / "agentlab.yaml"
    config_path.write_text("max_retries: -1\n", encoding="utf-8")

    with pytest.raises(CLIConfigError, match="max_retries"):
        load_cli_config(config_path)
