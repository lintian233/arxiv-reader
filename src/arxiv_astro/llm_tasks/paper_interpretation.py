from __future__ import annotations

import json
from typing import Any

from arxiv_astro.llm_client import LLMClient
from arxiv_astro.llm_tasks.base import LLMTask, LLMTaskResult
from arxiv_astro.models import LLMInterpretation, PaperMetadata


class PaperInterpretationTask(LLMTask[LLMInterpretation]):
    task_name = "paper_interpretation"
    prompt_version = "v1"
    schema_version = "v1"
    response_model = LLMInterpretation

    def run(
        self,
        llm_client: LLMClient,
        paper: PaperMetadata,
        text: str,
        max_input_chars: int,
    ) -> LLMTaskResult[LLMInterpretation]:
        raw = llm_client.chat_json(self.messages(paper, text))
        return LLMTaskResult(
            value=self.parse(raw),
            metadata=self.metadata(llm_client, max_input_chars),
        )

    def messages(self, paper: PaperMetadata, text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt(paper, text)},
        ]

    def parse(self, raw: str | dict[str, Any]) -> LLMInterpretation:
        return parse_interpretation(raw)


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


def parse_interpretation(raw_content: str | dict[str, Any]) -> LLMInterpretation:
    data = raw_content if isinstance(raw_content, dict) else json.loads(raw_content)
    data = normalize_interpretation_payload(data)
    return LLMInterpretation.model_validate(data)


def normalize_interpretation_payload(data: dict[str, Any]) -> dict[str, Any]:
    return dict(data)
