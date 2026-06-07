from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from arxiv_astro.http_client import create_http_client
from arxiv_astro.models import LLMInterpretation, PaperMetadata


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        client: Any | None = None,
        timeout: float = 60.0,
        reasoning_effort: str = "high",
        thinking_enabled: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("LLM API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.thinking_enabled = thinking_enabled
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
            http_client=create_http_client(timeout=timeout),
        )

    def interpret(self, paper: PaperMetadata, text: str) -> LLMInterpretation:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": user_prompt(paper, text)},
            ],
            stream=False,
            reasoning_effort=self.reasoning_effort,
            extra_body=extra_body(self.thinking_enabled),
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return parse_interpretation(content)


def system_prompt() -> str:
    return (
        "你是一个天文学论文解读助手。只返回合法 JSON，不要 Markdown。"
        "必须严格使用以下 JSON 结构和字段名："
        "{"
        '"one_sentence":"一句话说明论文做了什么",'
        '"background":"研究背景",'
        '"problem":"论文要解决的问题",'
        '"method":"核心方法",'
        '"result":"主要结果",'
        '"importance":"为什么重要",'
        '"limitations":"局限性或需要谨慎理解的地方"'
        "}。"
        "不要添加额外字段。"
    )


def user_prompt(paper: PaperMetadata, text: str) -> str:
    return (
        f"论文标题: {paper.title}\n"
        f"作者: {', '.join(paper.authors)}\n"
        f"arXiv ID: {paper.arxiv_id}\n"
        f"主分类: {paper.primary_category}\n"
        f"全部分类: {', '.join(paper.categories)}\n"
        f"DOI: {paper.doi or 'N/A'}\n"
        f"期刊引用: {paper.journal_ref or 'N/A'}\n"
        f"备注: {paper.comment or 'N/A'}\n"
        f"摘要: {paper.summary}\n\n"
        f"用于解读的论文内容:\n{text}\n\n"
        "请用中文生成结构化论文解读。"
    )


def extra_body(thinking_enabled: bool) -> dict[str, Any]:
    if not thinking_enabled:
        return {}
    return {"thinking": {"type": "enabled"}}


def parse_interpretation(raw_content: str | dict[str, Any]) -> LLMInterpretation:
    data = raw_content if isinstance(raw_content, dict) else json.loads(raw_content)
    data = normalize_interpretation_payload(data)
    return LLMInterpretation.model_validate(data)


def normalize_interpretation_payload(data: dict[str, Any]) -> dict[str, Any]:
    return dict(data)
