from __future__ import annotations

from collections.abc import Mapping

from agentlab.models.base import BaseModel, LLMMessage, ModelResponse


class MockModel(BaseModel):
    def __init__(
        self,
        default_response: str = "mock response",
        keyword_responses: Mapping[str, str] | None = None,
    ) -> None:
        self.default_response = default_response
        self.keyword_responses = dict(keyword_responses or {})

    def generate(self, messages: list[LLMMessage]) -> ModelResponse:
        content = _extract_latest_content(messages).lower()

        for keyword, response in self.keyword_responses.items():
            if keyword.lower() in content:
                return ModelResponse(content=response, model_name="mock")

        return ModelResponse(content=self.default_response, model_name="mock")


def _extract_latest_content(messages: list[LLMMessage]) -> str:
    if not messages:
        return ""
    return messages[-1].content
