from __future__ import annotations

from urllib.error import URLError

import pytest

import agentlab.cli as cli_module
from agentlab.cli import main, parse_args
from agentlab.multi_agent.supervisor import SupervisorConfig, build_default_supervisor


def test_parse_args_defaults() -> None:
    args = parse_args([])

    assert args.config is None
    assert args.use_openai is False
    assert args.output_dir is None
    assert args.search_mode is None
    assert args.no_search_fallback is False
    assert args.search_providers is None
    assert args.critic_mode is None
    assert args.strategy_preset is None
    assert args.list_strategy_presets is False
    assert args.print_effective_config is False
    assert args.max_retries is None
    assert args.agent_timeout_s is None
    assert args.retry_backoff_s is None
    assert args.retry_on_timeout_only is False
    assert args.continue_on_error is False
    assert args.topic is None


def test_parse_args_openai_and_output_dir() -> None:
    args = parse_args(
        [
            "test topic",
            "--use-openai",
            "--output-dir",
            "tmp_out",
            "--search-mode",
            "real",
            "--no-search-fallback",
            "--search-providers",
            "tavily,duckduckgo",
            "--critic-mode",
            "llm",
            "--strategy-preset",
            "concise",
            "--list-strategy-presets",
            "--print-effective-config",
            "--max-retries",
            "2",
            "--agent-timeout-s",
            "7.5",
            "--retry-backoff-s",
            "0.1",
            "--retry-on-timeout-only",
            "--continue-on-error",
        ]
    )

    assert args.topic == "test topic"
    assert args.use_openai is True
    assert args.output_dir == "tmp_out"
    assert args.search_mode == "real"
    assert args.no_search_fallback is True
    assert args.search_providers == "tavily,duckduckgo"
    assert args.critic_mode == "llm"
    assert args.strategy_preset == "concise"
    assert args.list_strategy_presets is True
    assert args.print_effective_config is True
    assert args.max_retries == 2
    assert args.agent_timeout_s == 7.5
    assert args.retry_backoff_s == 0.1
    assert args.retry_on_timeout_only is True
    assert args.continue_on_error is True


def test_main_supports_yaml_config_file(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "agentlab.yaml"
    config_path.write_text(
        "\n".join(
            [
                "topic: config topic",
                "output_dir: out_from_config",
                "search_mode: mock",
                "search_providers:",
                "  - duckduckgo",
                "  - wikipedia",
                "critic_mode: rule",
                "strategy_preset: concise",
                "no_search_fallback: false",
                "use_openai: false",
                "max_retries: 1",
                "agent_timeout_s: 5.0",
                "retry_backoff_s: 0.2",
                "retry_on_timeout_only: true",
                "continue_on_error: false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    assert (tmp_path / "out_from_config" / "report.md").exists()
    assert (tmp_path / "out_from_config" / "trace.json").exists()
    assert (tmp_path / "out_from_config" / "workspace.json").exists()
    assert (tmp_path / "out_from_config" / "run_summary.json").exists()


def test_main_cli_args_override_yaml_config(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "agentlab.yaml"
    config_path.write_text(
        "\n".join(
            [
                "topic: config topic",
                "output_dir: out_from_config",
                "strategy_preset: default",
                "search_mode: mock",
                "critic_mode: auto",
                "max_retries: 1",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "cli topic",
            "--config",
            str(config_path),
            "--output-dir",
            "out_from_cli",
            "--strategy-preset",
            "concise",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "out_from_cli" / "report.md").exists()
    assert not (tmp_path / "out_from_config" / "report.md").exists()


def test_main_runs_with_mock_mode(tmp_path) -> None:
    exit_code = main(["test topic", "--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "trace.json").exists()
    assert (tmp_path / "workspace.json").exists()
    assert (tmp_path / "run_summary.json").exists()


def test_main_runs_with_concise_strategy_preset(tmp_path) -> None:
    exit_code = main(
        [
            "test topic",
            "--output-dir",
            str(tmp_path),
            "--strategy-preset",
            "concise",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "trace.json").exists()
    assert (tmp_path / "workspace.json").exists()
    assert (tmp_path / "run_summary.json").exists()


def test_main_lists_strategy_presets_and_exits(capsys, tmp_path) -> None:
    exit_code = main(
        [
            "--list-strategy-presets",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Available strategy presets:" in out
    assert "- default:" in out
    assert "- concise:" in out
    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "trace.json").exists()
    assert not (tmp_path / "workspace.json").exists()
    assert not (tmp_path / "run_summary.json").exists()


def test_main_returns_config_error_for_missing_file(capsys) -> None:
    exit_code = main(["--config", "missing-agentlab.yaml"])

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "Config error:" in out


def test_main_prints_effective_config_and_exits(capsys, tmp_path) -> None:
    config_path = tmp_path / "agentlab.yaml"
    config_path.write_text(
        "\n".join(
            [
                "topic: config topic",
                "output_dir: out_from_config",
                "search_mode: mock",
                "search_providers: tavily,wikipedia",
                "critic_mode: rule",
                "strategy_preset: concise",
                "no_search_fallback: true",
                "use_openai: false",
                "max_retries: 1",
                "agent_timeout_s: 5.0",
                "retry_backoff_s: 0.2",
                "retry_on_timeout_only: true",
                "continue_on_error: true",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            "cli_out",
            "--print-effective-config",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert '"topic": "config topic"' in out
    assert '"output_dir": "cli_out"' in out
    assert '"search_mode": "mock"' in out
    assert '"strategy_preset": "concise"' in out
    assert '"no_search_fallback": true' in out
    assert '"max_retries": 1' in out
    assert '"agent_timeout_s": 5.0' in out
    assert '"retry_backoff_s": 0.2' in out
    assert '"retry_on_timeout_only": true' in out
    assert '"continue_on_error": true' in out


def test_main_returns_config_error_for_invalid_run_policy_args(capsys) -> None:
    exit_code = main(["--max-retries", "-1"])

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "Config error:" in out


def test_main_help_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Run AgentLab deep research demo." in out


def test_cli_entrypoint_uses_main_exit_code(monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "main", lambda argv=None: 7)

    with pytest.raises(SystemExit) as excinfo:
        cli_module.entrypoint()

    assert excinfo.value.code == 7


def test_supervisor_openai_mode_failure_still_exports_artifacts(monkeypatch, tmp_path) -> None:
    from agentlab.models.openai_compatible import OpenAICompatibleModel

    def _raise_model_error(self, messages):  # type: ignore[no-untyped-def]
        raise RuntimeError("OPENAI_API_KEY is not set")

    monkeypatch.setattr(OpenAICompatibleModel, "generate", _raise_model_error)
    monkeypatch.chdir(tmp_path)

    supervisor = build_default_supervisor(
        config=SupervisorConfig(
            output_dir=tmp_path / "out",
            use_openai_model=True,
        )
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        supervisor.run("Research LangGraph AutoGen CrewAI differences")

    assert (tmp_path / "out" / "report.md").exists()
    assert (tmp_path / "out" / "trace.json").exists()
    assert (tmp_path / "out" / "workspace.json").exists()
    assert (tmp_path / "out" / "run_summary.json").exists()


def test_supervisor_real_search_strict_mode_exports_artifacts_on_failure(
    monkeypatch, tmp_path
) -> None:
    import agentlab.tools.web_search as web_search

    def _fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        raise URLError("network down")

    monkeypatch.setattr(web_search, "urlopen", _fake_urlopen)
    monkeypatch.chdir(tmp_path)

    supervisor = build_default_supervisor(
        config=SupervisorConfig(
            output_dir=tmp_path / "strict_out",
            search_mode="real",
            allow_search_fallback=False,
        )
    )

    with pytest.raises(RuntimeError, match="duckduckgo_error|wikipedia_error"):
        supervisor.run("Research LangGraph AutoGen CrewAI differences")

    assert (tmp_path / "strict_out" / "report.md").exists()
    assert (tmp_path / "strict_out" / "trace.json").exists()
    assert (tmp_path / "strict_out" / "workspace.json").exists()
    assert (tmp_path / "strict_out" / "run_summary.json").exists()
