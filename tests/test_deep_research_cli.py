from __future__ import annotations

import importlib.util
from pathlib import Path
from urllib.error import URLError

import pytest

from agentlab.multi_agent.supervisor import build_default_supervisor


def _load_example_module():
    file_path = Path(__file__).resolve().parents[1] / "examples" / "04_deep_research.py"
    spec = importlib.util.spec_from_file_location("example_deep_research", file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load examples/04_deep_research.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults() -> None:
    module = _load_example_module()

    args = module.parse_args([])

    assert args.use_openai is False
    assert args.output_dir == "outputs"
    assert args.search_mode == "mock"
    assert args.no_search_fallback is False
    assert args.search_providers == "duckduckgo,wikipedia,tavily"
    assert "LangGraph" in args.topic


def test_parse_args_openai_and_output_dir() -> None:
    module = _load_example_module()

    args = module.parse_args(
        [
            "测试主题",
            "--use-openai",
            "--output-dir",
            "tmp_out",
            "--search-mode",
            "real",
            "--no-search-fallback",
            "--search-providers",
            "tavily,duckduckgo",
        ]
    )

    assert args.topic == "测试主题"
    assert args.use_openai is True
    assert args.output_dir == "tmp_out"
    assert args.search_mode == "real"
    assert args.no_search_fallback is True
    assert args.search_providers == "tavily,duckduckgo"


def test_main_runs_with_mock_mode(tmp_path) -> None:
    module = _load_example_module()

    exit_code = module.main(["测试主题", "--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "trace.json").exists()
    assert (tmp_path / "workspace.json").exists()


def test_supervisor_openai_mode_failure_still_exports_artifacts(monkeypatch, tmp_path) -> None:
    from agentlab.models.openai_compatible import OpenAICompatibleModel

    def _raise_model_error(self, messages):  # type: ignore[no-untyped-def]
        raise RuntimeError("OPENAI_API_KEY is not set")

    monkeypatch.setattr(OpenAICompatibleModel, "generate", _raise_model_error)
    monkeypatch.chdir(tmp_path)

    supervisor = build_default_supervisor(
        output_dir=tmp_path / "out",
        use_openai_model=True,
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        supervisor.run("研究 LangGraph、AutoGen、CrewAI 的区别")

    assert (tmp_path / "out" / "report.md").exists()
    assert (tmp_path / "out" / "trace.json").exists()
    assert (tmp_path / "out" / "workspace.json").exists()


def test_supervisor_real_search_strict_mode_exports_artifacts_on_failure(
    monkeypatch, tmp_path
) -> None:
    import agentlab.tools.web_search as web_search

    def _fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        raise URLError("network down")

    monkeypatch.setattr(web_search, "urlopen", _fake_urlopen)
    monkeypatch.chdir(tmp_path)

    supervisor = build_default_supervisor(
        output_dir=tmp_path / "strict_out",
        search_mode="real",
        allow_search_fallback=False,
    )

    with pytest.raises(RuntimeError, match="duckduckgo_error|wikipedia_error"):
        supervisor.run("研究 LangGraph、AutoGen、CrewAI 的区别")

    assert (tmp_path / "strict_out" / "report.md").exists()
    assert (tmp_path / "strict_out" / "trace.json").exists()
    assert (tmp_path / "strict_out" / "workspace.json").exists()
