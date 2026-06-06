from __future__ import annotations

import json
from typing import Any

import httpx

from arxiv_astro.http_client import create_http_client
from arxiv_astro.models import LLMInterpretation, PaperMetadata


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        http_client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("LLM API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = http_client or create_http_client(timeout=timeout)

    def interpret(self, paper: PaperMetadata, text: str) -> LLMInterpretation:
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": user_prompt(paper, text)},
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        return parse_interpretation(content)


def system_prompt() -> str:
    return (
        "你是一个天文学论文解读助手。只返回合法 JSON，不要 Markdown。"
        "字段必须包含 one_sentence, background, problem, method, result, "
        "importance, limitations, keywords, reading_level, recommended_for。"
    )


def user_prompt(paper: PaperMetadata, text: str) -> str:
    return (
        f"论文标题: {paper.title}\n"
        f"作者: {', '.join(paper.authors)}\n"
        f"arXiv ID: {paper.arxiv_id}\n"
        f"分类: {', '.join(paper.categories)}\n"
        f"摘要: {paper.summary}\n\n"
        f"正文片段:\n{text}\n\n"
        "请用中文生成结构化论文解读。"
    )


def parse_interpretation(raw_content: str | dict[str, Any]) -> LLMInterpretation:
    data = raw_content if isinstance(raw_content, dict) else json.loads(raw_content)
    return LLMInterpretation.model_validate(data)
