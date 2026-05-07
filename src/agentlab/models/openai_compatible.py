from __future__ import annotations

import os
from typing import Any, cast

from dotenv import load_dotenv
from openai import OpenAI

from agentlab.models.base import BaseModel


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

    def generate(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Use MockModel for offline runs.")

        client = self._client
        if client is None:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self._client = client

        model_name = self.model if isinstance(self.model, str) else "gpt-4o-mini"
        response = client.chat.completions.create(
            model=cast(Any, model_name),
            messages=cast(Any, messages),
        )
        content = response.choices[0].message.content
        return content or ""
