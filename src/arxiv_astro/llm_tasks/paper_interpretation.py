from __future__ import annotations

import json
from typing import Any

from arxiv_astro.llm_client import LLMClient
from arxiv_astro.llm_tasks.base import LLMTask, LLMTaskResult
from arxiv_astro.models import LLMInterpretation, PaperMetadata


class PaperInterpretationTask(LLMTask[LLMInterpretation]):
    task_name = "paper_interpretation"
    prompt_version = "v2"
    schema_version = "v2"
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
        "你是一个天文学论文解读助手。读者可能不熟悉论文所属子领域，"
        "你的目标是帮助读者理解这篇论文解决了什么前沿问题、主流方法是什么、结果意味着什么。"
        "只返回合法 JSON，不要 Markdown，不要添加额外字段。"
        "所有字段都用中文，表达要直白、具体、避免空泛评价。"
        "字段输出要求："
        "one_sentence 用一句话概括论文的科学或技术贡献；"
        "problem_context 说明论文面对的科学问题或技术问题，以及这个问题来自什么背景；"
        "why_it_matters 说明为什么这个问题值得做，最好联系观测、理论、仪器、数据或方法上的实际需求；"
        "what_the_paper_does 说明论文具体做了什么，包括数据、模型、实验、观测、模拟或算法；"
        "main_results 说明最核心结果，不要只说“效果很好”，要写出可理解的结论、指标或趋势；"
        "key_figures 最多 2 项，每项必须引用输入中的图片编号 index，index 从 1 开始，不能编造没有出现的图片；"
        "key_figures.plain_caption 用最直白的话说明这张图在展示什么；"
        "key_figures.why_key 说明为什么这张图关键；"
        "key_figures.evidence 如果图片说明或正文中有具体数值、对象、趋势，就写进去，否则为 null；"
        "如果输入图片信息不足以判断关键图，key_figures 返回空数组；"
        "limitations 说明论文局限、假设、适用范围或需要谨慎理解的地方；"
        "field_position 说明这篇论文放在该领域里的位置：它体现了什么主流方法、前沿趋势，或者可能往哪里发展。"
        "必须严格使用以下 JSON 骨架："
        "{"
        '"one_sentence":"",'
        '"problem_context":"",'
        '"why_it_matters":"",'
        '"what_the_paper_does":"",'
        '"main_results":"",'
        '"key_figures":[{"index":1,"plain_caption":"","why_key":"","evidence":null}],'
        '"limitations":"",'
        '"field_position":""'
        "}。"
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
