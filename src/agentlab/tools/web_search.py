from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from agentlab.tools.base import BaseTool


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search tool with mock/real modes for deep-research demos."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(
        self,
        mode: str = "mock",
        timeout_s: float = 8.0,
        allow_fallback: bool = True,
    ) -> None:
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"mock", "real"}:
            raise ValueError("mode must be 'mock' or 'real'")
        self.mode = normalized_mode
        self.timeout_s = timeout_s
        self.allow_fallback = allow_fallback

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
            except (URLError, TimeoutError, ValueError, OSError) as exc:
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
        fallback["query_candidates"] = candidates
        return fallback

    def _run_real_search(self, query: str) -> dict[str, Any]:
        results: list[dict[str, str]] = []
        source_hits = {"duckduckgo": 0, "wikipedia": 0}
        issues: list[str] = []

        try:
            ddg_results = self._search_duckduckgo(query)
            source_hits["duckduckgo"] = len(ddg_results)
            results.extend(ddg_results)
        except (URLError, TimeoutError, ValueError, OSError) as exc:
            issues.append(f"duckduckgo_error: {exc}")

        try:
            wiki_results = self._search_wikipedia(query)
            source_hits["wikipedia"] = len(wiki_results)
            results.extend(_deduplicate_results(results, wiki_results))
        except (URLError, TimeoutError, ValueError, OSError) as exc:
            issues.append(f"wikipedia_error: {exc}")

        return {
            "query": query,
            "results": results,
            "mode": "real",
            "fallback_used": False,
            "source_hits": source_hits,
            "real_issues": issues,
        }

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


def _contains_cjk(text: str) -> bool:
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


def _mock_search(query: str) -> dict[str, Any]:
    normalized = query.lower()
    results: list[dict[str, str]] = []

    if "langgraph" in normalized:
        results.append(
            {
                "title": "LangGraph overview",
                "snippet": (
                    "Uses graph/state-machine style orchestration with explicit nodes, "
                    "edges, and durable state for complex multi-step workflows."
                ),
                "source": "mock://langgraph/overview",
            }
        )
    if "autogen" in normalized:
        results.append(
            {
                "title": "AutoGen overview",
                "snippet": (
                    "Centers on multi-agent conversation loops and message-driven collaboration, "
                    "convenient for dialogue-heavy automation."
                ),
                "source": "mock://autogen/overview",
            }
        )
    if "crewai" in normalized:
        results.append(
            {
                "title": "CrewAI overview",
                "snippet": (
                    "Emphasizes role-based crews and task delegation, often easier to start "
                    "for team/workflow style business use cases."
                ),
                "source": "mock://crewai/overview",
            }
        )
    if "langgraph" in normalized and "autogen" in normalized and "crewai" in normalized:
        results.append(
            {
                "title": "Framework comparison snapshot",
                "snippet": (
                    "LangGraph: strongest for explicit stateful orchestration; "
                    "AutoGen: strongest for conversational agent collaboration; "
                    "CrewAI: strongest for role/task-driven execution speed."
                ),
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
    }


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
