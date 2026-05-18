from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.error import URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from agentlab.tools.base import BaseTool


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search tool with mock/real modes for deep-research demos."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    def __init__(
        self,
        mode: str = "mock",
        timeout_s: float = 8.0,
        allow_fallback: bool = True,
        real_providers: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"mock", "real"}:
            raise ValueError("mode must be 'mock' or 'real'")
        self.mode = normalized_mode
        self.timeout_s = timeout_s
        self.allow_fallback = allow_fallback
        self.real_providers = _normalize_real_providers(real_providers)

    def run(self, **kwargs: Any) -> dict[str, Any]:
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("'query' must be a non-empty string")

        cleaned_query = query.strip()
        if self.mode == "mock":
            return _mock_search(cleaned_query)

        candidates = _build_real_query_candidates(cleaned_query)
        errors: list[str] = []

        for candidate in candidates:
            try:
                real_result = self._run_real_search(candidate)
            except (URLError, TimeoutError, ValueError, OSError, RuntimeError) as exc:
                errors.append(f"{candidate}: real_search_error: {exc}")
                continue

            if real_result["results"]:
                real_result["query_original"] = cleaned_query
                real_result["query_used"] = candidate
                real_result["query_candidates"] = candidates
                return real_result

            issues = real_result.get("real_issues", [])
            if isinstance(issues, list) and issues:
                errors.extend([f"{candidate}: {issue}" for issue in issues])
            else:
                errors.append(f"{candidate}: real_search_empty")

        if not self.allow_fallback:
            if errors and any("_error:" in item for item in errors):
                raise RuntimeError("; ".join(errors))
            raise RuntimeError(
                "real_search_empty "
                f"(candidates tried: {', '.join(candidates)})"
            )

        fallback = _mock_search(cleaned_query)
        fallback["fallback_used"] = True
        fallback["fallback_reason"] = errors[0] if errors else "real_search_empty"
        fallback["real_issues"] = errors
        fallback["query_candidates"] = candidates
        return fallback

    def _run_real_search(self, query: str) -> dict[str, Any]:
        results: list[dict[str, str]] = []
        source_hits: dict[str, int] = {provider: 0 for provider in self.real_providers}
        issues: list[str] = []

        for provider in self.real_providers:
            try:
                provider_results = self._search_by_provider(provider, query)
            except (URLError, TimeoutError, ValueError, OSError, RuntimeError) as exc:
                issues.append(f"{provider}_error: {exc}")
                continue

            source_hits[provider] = len(provider_results)
            if provider_results:
                results.extend(_deduplicate_results(results, provider_results))

        return {
            "query": query,
            "results": results,
            "mode": "real",
            "fallback_used": False,
            "source_hits": source_hits,
            "real_issues": issues,
        }

    def _search_by_provider(self, provider: str, query: str) -> list[dict[str, str]]:
        if provider == "duckduckgo":
            return self._search_duckduckgo(query)
        if provider == "wikipedia":
            return self._search_wikipedia(query)
        if provider == "tavily":
            return self._search_tavily(query)
        raise ValueError(f"unsupported_provider: {provider}")

    def _search_duckduckgo(self, query: str) -> list[dict[str, str]]:
        endpoint = (
            "https://api.duckduckgo.com/"
            f"?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
        )
        request = Request(endpoint, headers={"User-Agent": "AgentLab/0.1"})
        with urlopen(request, timeout=self.timeout_s) as response:
            payload = json.load(response)

        results: list[dict[str, str]] = []

        abstract_text = str(payload.get("AbstractText", "")).strip()
        abstract_url = str(payload.get("AbstractURL", "")).strip()
        heading = str(payload.get("Heading", "")).strip()
        if abstract_text:
            results.append(
                {
                    "title": heading or "DuckDuckGo abstract",
                    "snippet": abstract_text,
                    "source": abstract_url or "https://duckduckgo.com",
                }
            )

        related_topics = payload.get("RelatedTopics", [])
        if isinstance(related_topics, list):
            for item in _flatten_related_topics(related_topics):
                text = str(item.get("Text", "")).strip()
                source = str(item.get("FirstURL", "")).strip()
                if not text:
                    continue
                title, snippet = _split_topic_text(text)
                results.append(
                    {
                        "title": title,
                        "snippet": snippet,
                        "source": source or "https://duckduckgo.com",
                    }
                )
                if len(results) >= 6:
                    break

        return results

    def _search_tavily(self, query: str) -> list[dict[str, str]]:
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            raise ValueError("tavily_missing_api_key")

        base_url = os.getenv("TAVILY_BASE_URL", "https://api.tavily.com").strip()
        endpoint = f"{base_url.rstrip('/')}/search"
        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
            "include_answer": True,
            "include_raw_content": False,
        }
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "AgentLab/0.1",
            },
            method="POST",
        )

        with urlopen(request, timeout=self.timeout_s) as response:
            raw_payload = response.read()
        data = json.loads(raw_payload.decode("utf-8"))

        results: list[dict[str, str]] = []
        answer = str(data.get("answer", "")).strip()
        if answer:
            results.append(
                {
                    "title": "Tavily answer",
                    "snippet": answer,
                    "source": "https://api.tavily.com/search",
                }
            )

        source_items = data.get("results", [])
        if not isinstance(source_items, list):
            return results

        for item in source_items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip() or "Tavily result"
            snippet = str(item.get("content", "")).strip()
            url = str(item.get("url", "")).strip() or "https://tavily.com"
            if not snippet and not title:
                continue
            results.append({"title": title, "snippet": snippet or title, "source": url})

        return results

    def _search_wikipedia(self, query: str) -> list[dict[str, str]]:
        endpoint = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&format=json&srlimit=3&utf8=1&srsearch={quote_plus(query)}"
        )
        request = Request(endpoint, headers={"User-Agent": "AgentLab/0.1"})
        with urlopen(request, timeout=self.timeout_s) as response:
            payload = json.load(response)

        results: list[dict[str, str]] = []
        query_node = payload.get("query", {})
        search_items = query_node.get("search", []) if isinstance(query_node, dict) else []
        if not isinstance(search_items, list):
            return results

        for item in search_items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            snippet_raw = str(item.get("snippet", "")).strip()
            snippet = _strip_html_tags(snippet_raw)
            if not title and not snippet:
                continue
            page_title = quote_plus(title.replace(" ", "_")) if title else quote_plus(query)
            results.append(
                {
                    "title": title or "Wikipedia search result",
                    "snippet": snippet or "Wikipedia matched this query.",
                    "source": f"https://en.wikipedia.org/wiki/{page_title}",
                }
            )

        return results


def _build_real_query_candidates(query: str) -> list[str]:
    candidates: list[str] = [query]

    lowered = query.lower()
    framework_alias = {
        "langgraph": "LangGraph",
        "autogen": "AutoGen",
        "crewai": "CrewAI",
    }
    framework_tokens: list[str] = []
    for keyword in ["langgraph", "autogen", "crewai"]:
        if keyword in lowered:
            framework_tokens.append(framework_alias[keyword])

    if framework_tokens:
        framework_phrase = " ".join(framework_tokens)
        candidates.append(f"{framework_phrase} comparison")
        candidates.append(f"{framework_phrase} framework differences")
        candidates.append(framework_phrase)

    ascii_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.-]*", query)
    if ascii_tokens:
        candidates.append(" ".join(dict.fromkeys(ascii_tokens)))

    if _contains_cjk(query) and framework_tokens:
        candidates.append(f"Compare {' '.join(framework_tokens)}")

    return list(dict.fromkeys(item.strip() for item in candidates if item.strip()))


def _normalize_real_providers(
    providers: list[str] | tuple[str, ...] | None,
) -> list[str]:
    default_providers = ["duckduckgo", "wikipedia", "tavily"]
    selected = providers if providers is not None else default_providers

    normalized: list[str] = []
    for item in selected:
        name = item.strip().lower()
        if not name:
            continue
        if name not in {"duckduckgo", "wikipedia", "tavily"}:
            raise ValueError(
                "real providers must be any of: duckduckgo, wikipedia, tavily"
            )
        normalized.append(name)

    deduplicated = list(dict.fromkeys(normalized))
    if not deduplicated:
        raise ValueError("real providers cannot be empty")
    return deduplicated


def _contains_cjk(text: str) -> bool:
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


_MOCK_DIMENSIONS = (
    "core_paradigm",
    "coordination_style",
    "state_memory",
    "best_fit",
    "trade_off",
)

_MOCK_DIMENSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "core_paradigm": ("核心设计", "理念", "目标场景", "core", "paradigm"),
    "coordination_style": ("任务编排", "对话", "角色分工", "coordination", "workflow", "orchestration"),
    "state_memory": ("状态管理", "工具调用", "可观测", "state", "memory", "observability"),
    "best_fit": ("技术选型", "团队", "决策", "best fit", "selection", "scenario"),
    "trade_off": ("优缺点", "学习成本", "扩展性", "工程落地", "trade-off", "cost", "risk"),
}

_MOCK_FRAMEWORK_SNIPPETS: dict[str, dict[str, str]] = {
    "core_paradigm": {
        "langgraph": "LangGraph uses graph/state-machine orchestration for explicit multi-step control.",
        "autogen": "AutoGen focuses on conversation-centric multi-agent collaboration loops.",
        "crewai": "CrewAI emphasizes role-task crews for practical business workflow automation.",
    },
    "coordination_style": {
        "langgraph": "LangGraph coordinates via deterministic node-edge transitions with explicit routing.",
        "autogen": "AutoGen coordinates by turn-based dialogue and message passing between agents.",
        "crewai": "CrewAI coordinates through role delegation, task assignment, and crew planning.",
    },
    "state_memory": {
        "langgraph": "LangGraph keeps explicit workflow state and supports checkpoint-style recovery.",
        "autogen": "AutoGen relies on conversation history; long-term memory is often external.",
        "crewai": "CrewAI keeps lightweight shared context around roles, goals, and task outputs.",
    },
    "best_fit": {
        "langgraph": "LangGraph best fits complex, auditable, long-running production workflows.",
        "autogen": "AutoGen best fits rapid prototyping of collaborative conversational agents.",
        "crewai": "CrewAI best fits business process automation with clear role ownership.",
    },
    "trade_off": {
        "langgraph": "LangGraph trade-off: strong control and observability with higher setup complexity.",
        "autogen": "AutoGen trade-off: flexible interaction but easier to drift without constraints.",
        "crewai": "CrewAI trade-off: fast onboarding but limited deep customization in some scenarios.",
    },
}

_MOCK_COMPARISON_SNIPPETS: dict[str, str] = {
    "core_paradigm": (
        "Comparison: LangGraph favors explicit graph control, AutoGen favors conversational loops, "
        "CrewAI favors role-task workflows."
    ),
    "coordination_style": (
        "Coordination comparison: LangGraph uses deterministic graph routing, AutoGen uses dialogue turns, "
        "CrewAI uses role delegation."
    ),
    "state_memory": (
        "State comparison: LangGraph has explicit state checkpoints, AutoGen centers on chat history, "
        "CrewAI uses lightweight shared context."
    ),
    "best_fit": (
        "Scenario comparison: LangGraph for complex production orchestration, AutoGen for experiments, "
        "CrewAI for business automation."
    ),
    "trade_off": (
        "Trade-off comparison: LangGraph has higher complexity, AutoGen risks drift, "
        "CrewAI may limit deep customization."
    ),
}


def _mock_search(query: str) -> dict[str, Any]:
    normalized = query.lower()
    dimension = _detect_mock_dimension(query)
    results: list[dict[str, str]] = []
    snippets = _MOCK_FRAMEWORK_SNIPPETS.get(dimension, _MOCK_FRAMEWORK_SNIPPETS["core_paradigm"])

    if "langgraph" in normalized:
        results.append(
            {
                "title": f"LangGraph {dimension}",
                "snippet": snippets["langgraph"],
                "source": "mock://langgraph/overview",
            }
        )
    if "autogen" in normalized:
        results.append(
            {
                "title": f"AutoGen {dimension}",
                "snippet": snippets["autogen"],
                "source": "mock://autogen/overview",
            }
        )
    if "crewai" in normalized:
        results.append(
            {
                "title": f"CrewAI {dimension}",
                "snippet": snippets["crewai"],
                "source": "mock://crewai/overview",
            }
        )
    if "langgraph" in normalized and "autogen" in normalized and "crewai" in normalized:
        results.append(
            {
                "title": f"Framework comparison snapshot ({dimension})",
                "snippet": _MOCK_COMPARISON_SNIPPETS.get(dimension, _MOCK_COMPARISON_SNIPPETS["core_paradigm"]),
                "source": "mock://comparison/langgraph-autogen-crewai",
            }
        )

    if not results:
        results.append(
            {
                "title": "General research result",
                "snippet": f"No specific mock corpus for '{query}', returning generic context.",
                "source": "mock://general/result",
            }
        )

    return {
        "query": query,
        "results": results,
        "mode": "mock",
        "fallback_used": False,
        "mock_dimension": dimension,
    }


def _detect_mock_dimension(query: str) -> str:
    lowered = query.lower()
    for dimension in _MOCK_DIMENSIONS:
        keywords = _MOCK_DIMENSION_KEYWORDS.get(dimension, ())
        if any(keyword.lower() in lowered for keyword in keywords):
            return dimension
    return "core_paradigm"


def _flatten_related_topics(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "Topics" in item and isinstance(item.get("Topics"), list):
            nested = item.get("Topics")
            if isinstance(nested, list):
                for child in nested:
                    if isinstance(child, dict):
                        flat.append(child)
        else:
            flat.append(item)
    return flat


def _split_topic_text(text: str) -> tuple[str, str]:
    if " - " in text:
        title, snippet = text.split(" - ", maxsplit=1)
        return title.strip(), snippet.strip()
    return text, text


def _strip_html_tags(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", "", text)
    return cleaned.replace("&quot;", "\"").replace("&#39;", "'").strip()


def _deduplicate_results(
    existing: list[dict[str, str]], new_results: list[dict[str, str]]
) -> list[dict[str, str]]:
    seen = {
        (
            str(item.get("title", "")).strip().lower(),
            str(item.get("source", "")).strip().lower(),
        )
        for item in existing
    }
    merged: list[dict[str, str]] = []
    for item in new_results:
        key = (
            str(item.get("title", "")).strip().lower(),
            str(item.get("source", "")).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged
