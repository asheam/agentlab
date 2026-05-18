from __future__ import annotations

from urllib.error import URLError

import pytest

from agentlab.cli import main, parse_args
from agentlab.multi_agent.supervisor import SupervisorConfig, build_default_supervisor


def test_parse_args_defaults() -> None:
    args = parse_args([])

    assert args.use_openai is False
    assert args.output_dir == "outputs"
    assert args.search_mode == "mock"
    assert args.no_search_fallback is False
    assert args.search_providers == "duckduckgo,wikipedia,tavily"
    assert args.critic_mode == "auto"
    assert args.strategy_preset == "default"
    assert args.list_strategy_presets is False
    assert "LangGraph" in args.topic


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
