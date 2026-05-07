from __future__ import annotations

from typing import Any

from agentlab.tools.base import BaseTool


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Mock web search for offline deep-research demos."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def run(self, **kwargs: Any) -> dict[str, Any]:
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("'query' must be a non-empty string")

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
        if (
            "langgraph" in normalized
            and "autogen" in normalized
            and "crewai" in normalized
        ):
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

        return {"query": query, "results": results}
