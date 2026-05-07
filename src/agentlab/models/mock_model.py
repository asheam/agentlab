from __future__ import annotations

from collections.abc import Mapping

from agentlab.models.base import BaseModel


class MockModel(BaseModel):
    def __init__(
        self,
        default_response: str = "mock response",
        keyword_responses: Mapping[str, str] | None = None,
    ) -> None:
        self.default_response = default_response
        self.keyword_responses = dict(keyword_responses or {})

    def generate(self, messages: list[dict[str, str]]) -> str:
        content = _extract_latest_content(messages).lower()

        for keyword, response in self.keyword_responses.items():
            if keyword.lower() in content:
                return response

        return self.default_response


def _extract_latest_content(messages: list[dict[str, str]]) -> str:
    if not messages:
        return ""
    return messages[-1].get("content", "")
