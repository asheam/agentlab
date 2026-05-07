from __future__ import annotations

import io
import json
from urllib.error import URLError

import agentlab.tools.web_search as web_search
from agentlab.tools.web_search import WebSearchTool


class _FakeHTTPResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def test_web_search_mock_mode_returns_mock_results() -> None:
    tool = WebSearchTool(mode="mock")

    result = tool.run(query="LangGraph AutoGen CrewAI")

    assert result["mode"] == "mock"
    assert result["fallback_used"] is False
    sources = [item["source"] for item in result["results"]]
    assert "mock://langgraph/overview" in sources
    assert "mock://autogen/overview" in sources
    assert "mock://crewai/overview" in sources


def test_web_search_real_mode_returns_real_results(monkeypatch) -> None:
    payload = {
        "Heading": "Test Heading",
        "AbstractText": "Test abstract",
        "AbstractURL": "https://example.com/abstract",
        "RelatedTopics": [
            {"Text": "Topic A - Snippet A", "FirstURL": "https://example.com/a"},
            {"Text": "Topic B", "FirstURL": "https://example.com/b"},
        ],
    }

    def _fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        return _FakeHTTPResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(web_search, "urlopen", _fake_urlopen)

    tool = WebSearchTool(mode="real")
    result = tool.run(query="test query")

    assert result["mode"] == "real"
    assert result["fallback_used"] is False
    assert len(result["results"]) >= 1
    assert result["results"][0]["source"].startswith("https://")
    assert result["source_hits"]["duckduckgo"] >= 1
    assert "wikipedia" in result["source_hits"]


def test_web_search_real_mode_falls_back_to_mock_on_error(monkeypatch) -> None:
    def _fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        raise URLError("network down")

    monkeypatch.setattr(web_search, "urlopen", _fake_urlopen)

    tool = WebSearchTool(mode="real")
    result = tool.run(query="LangGraph")

    assert result["mode"] == "mock"
    assert result["fallback_used"] is True
    fallback_reason = result.get("fallback_reason", "")
    assert "duckduckgo_error" in fallback_reason or "wikipedia_error" in fallback_reason
    sources = [item["source"] for item in result["results"]]
    assert "mock://langgraph/overview" in sources


def test_web_search_real_mode_raises_when_fallback_disabled(monkeypatch) -> None:
    def _fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        raise URLError("network down")

    monkeypatch.setattr(web_search, "urlopen", _fake_urlopen)

    tool = WebSearchTool(mode="real", allow_fallback=False)

    import pytest

    with pytest.raises(RuntimeError, match="duckduckgo_error|wikipedia_error"):
        tool.run(query="LangGraph")


def test_web_search_real_mode_rewrites_query_and_can_succeed(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_run_real_search(self, query):  # type: ignore[no-untyped-def]
        calls.append(query)
        if query == "LangGraph AutoGen CrewAI comparison":
            return {
                "query": query,
                "results": [
                    {"title": "hit", "snippet": "rewritten query matched", "source": "https://example.com"}
                ],
                "mode": "real",
                "fallback_used": False,
                "source_hits": {"duckduckgo": 1, "wikipedia": 0},
                "real_issues": [],
            }
        return {
            "query": query,
            "results": [],
            "mode": "real",
            "fallback_used": False,
            "source_hits": {"duckduckgo": 0, "wikipedia": 0},
            "real_issues": ["real_search_empty"],
        }

    monkeypatch.setattr(WebSearchTool, "_run_real_search", _fake_run_real_search)

    tool = WebSearchTool(mode="real", allow_fallback=False)
    result = tool.run(query="研究 LangGraph、AutoGen、CrewAI 的区别")

    assert result["mode"] == "real"
    assert result["fallback_used"] is False
    assert result["query_used"] == "LangGraph AutoGen CrewAI comparison"
    assert result["query_original"] == "研究 LangGraph、AutoGen、CrewAI 的区别"
    assert "LangGraph AutoGen CrewAI comparison" in result["query_candidates"]
    assert calls[0] == "研究 LangGraph、AutoGen、CrewAI 的区别"
    assert "LangGraph AutoGen CrewAI comparison" in calls


def test_web_search_real_mode_uses_wikipedia_when_ddg_empty(monkeypatch) -> None:
    def _fake_ddg(self, query):  # type: ignore[no-untyped-def]
        return []

    def _fake_wiki(self, query):  # type: ignore[no-untyped-def]
        return [
            {
                "title": "LangGraph",
                "snippet": "Wikipedia hit",
                "source": "https://en.wikipedia.org/wiki/LangGraph",
            }
        ]

    monkeypatch.setattr(WebSearchTool, "_search_duckduckgo", _fake_ddg)
    monkeypatch.setattr(WebSearchTool, "_search_wikipedia", _fake_wiki)

    tool = WebSearchTool(mode="real", allow_fallback=False)
    result = tool.run(query="研究 LangGraph、AutoGen、CrewAI 的区别")

    assert result["mode"] == "real"
    assert result["fallback_used"] is False
    assert result["source_hits"]["duckduckgo"] == 0
    assert result["source_hits"]["wikipedia"] == 1
    assert result["results"][0]["source"].startswith("https://en.wikipedia.org/wiki/")
