from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from arxiv_astro.http_client import create_http_client


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        client: Any | None = None,
        timeout: float = 60.0,
        reasoning_effort: str = "high",
        thinking_enabled: bool = False,
        max_output_tokens: int = 12000,
    ) -> None:
        if not api_key:
            raise ValueError("LLM API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.thinking_enabled = thinking_enabled
        self.max_output_tokens = max_output_tokens
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
            http_client=create_http_client(timeout=timeout),
        )

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            reasoning_effort=self.reasoning_effort,
            extra_body=extra_body(self.thinking_enabled),
            response_format={"type": "json_object"},
            max_tokens=self.max_output_tokens,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            return dict(content)
        return json.loads(content)


def extra_body(thinking_enabled: bool) -> dict[str, Any]:
    if not thinking_enabled:
        return {}
    return {"thinking": {"type": "enabled"}}
