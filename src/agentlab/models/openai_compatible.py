from __future__ import annotations

import os
from typing import Any, cast

from dotenv import load_dotenv
from openai import OpenAI

from agentlab.models.base import BaseModel, LLMMessage, ModelResponse


class OpenAICompatibleModel(BaseModel):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client: OpenAI | None = None

    def generate(self, messages: list[LLMMessage]) -> ModelResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Use MockModel for offline runs.")

        client = self._client
        if client is None:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self._client = client

        model_name = self.model if isinstance(self.model, str) else "gpt-4o-mini"
        payload = [{"role": msg.role, "content": msg.content} for msg in messages]
        response = client.chat.completions.create(
            # OpenAI SDK typing uses complex overloads; cast keeps our adapter API strongly typed.
            model=cast(Any, model_name),
            # Payload shape is validated by our LLMMessage schema before this cast.
            messages=cast(Any, payload),
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        tokens_input = usage.prompt_tokens if usage is not None else 0
        tokens_output = usage.completion_tokens if usage is not None else 0
        finish_reason = response.choices[0].finish_reason or ""
        response_model_name = response.model or model_name

        return ModelResponse(
            content=content,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model_name=response_model_name,
            finish_reason=finish_reason,
        )
