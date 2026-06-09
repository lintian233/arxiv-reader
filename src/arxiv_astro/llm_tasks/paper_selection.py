from __future__ import annotations

import json
from typing import Any

from arxiv_astro.llm_client import LLMClient
from arxiv_astro.llm_tasks.base import LLMTask, LLMTaskResult
from arxiv_astro.models import PaperMetadata, PaperSelectionResult


DEFAULT_SUMMARY_MAX_CHARS = 4000


class PaperSelectionTask(LLMTask[PaperSelectionResult]):
    task_name = "paper_selection"
    prompt_version = "v1"
    schema_version = "v1"
    response_model = PaperSelectionResult

    def run(
        self,
        llm_client: LLMClient,
        papers: list[PaperMetadata],
        interests: str,
        max_results: int,
        max_input_chars: int,
        summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
    ) -> LLMTaskResult[PaperSelectionResult]:
        messages = self.messages(papers, interests, max_results, max_input_chars, summary_max_chars)
        raw = llm_client.chat_json(messages)
        return LLMTaskResult(
            value=self.parse(raw),
            metadata=self.metadata(llm_client, max_input_chars),
        )

    def messages(
        self,
        papers: list[PaperMetadata],
        interests: str,
        max_results: int,
        max_input_chars: int,
        summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
    ) -> list[dict[str, str]]:
        user_content = user_prompt(papers, interests, max_results, summary_max_chars)
        total_chars = len(system_prompt()) + len(user_content)
        if total_chars > max_input_chars:
            raise ValueError(
                "Selection input too long. Reduce --fetch-results or increase SELECTION_MAX_INPUT_CHARS."
            )
        return [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_content},
        ]

    def parse(self, raw: str | dict[str, Any]) -> PaperSelectionResult:
        data = raw if isinstance(raw, dict) else json.loads(raw)
        return PaperSelectionResult.model_validate(data)


def system_prompt() -> str:
    return (
        "你是一个论文筛选助手。你的任务是根据用户兴趣，从候选 arXiv 论文 metadata 中筛选最值得继续阅读全文的论文。"
        "只返回合法 JSON，不要 Markdown，不要添加额外字段。"
        "只能选择候选列表中真实存在的 arxiv_id，不能编造论文。"
        "最多返回用户要求的 max_results 篇论文。"
        "在不选择明显无关论文的前提下，请尽量返回接近 max_results 的论文。"
        "如果高度相关论文少于 max_results，可以补充选择中等相关但仍有阅读价值的论文；不要为了凑满选择弱相关论文。"
        "如果少选，必须在 shortfall_reason 中用一句话说明原因；如果选满，shortfall_reason 使用空字符串。"
        "不要只因为标题或摘要中出现关键词就机械选择，要结合摘要判断科学问题、方法和结果是否真的相关。"
        "优先选择与兴趣高度相关、贡献清楚、对理解领域前沿或主流方法有价值的论文。"
        "relevance 评分标准：5=高度相关且应优先阅读；4=明显相关且建议阅读；3=中等相关，强相关不足时可选；2=弱相关，不应选择；1=无关，不应选择。"
        "按相关性从高到低排序。"
        "必须严格使用以下 JSON 骨架："
        "{"
        '"selected":['
        '{"arxiv_id":"","relevance":5,"matched_interests":[""],"reason":""}'
        "],"
        '"shortfall_reason":""'
        "}。"
    )


def user_prompt(
    papers: list[PaperMetadata],
    interests: str,
    max_results: int,
    summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
) -> str:
    candidates = "\n\n".join(
        format_candidate(index, paper, summary_max_chars)
        for index, paper in enumerate(papers, start=1)
    )
    return (
        f"用户兴趣:\n{interests}\n\n"
        f"选择论文数 max_results: {max_results}\n\n"
        f"候选论文总数: {len(papers)}\n\n"
        f"候选论文 metadata:\n{candidates}\n\n"
        "请返回 JSON。reason 用一句话说明为什么该论文值得进入后续阅读流程。"
        "请优先选 relevance 5 和 4 的论文；如果数量不足，可以选择 relevance 3 但仍有助于理解领域前沿或主流方法的论文。"
        "如果 selected 数量少于 max_results，shortfall_reason 必须说明候选中相关论文不足的原因。"
    )


def format_candidate(index: int, paper: PaperMetadata, summary_max_chars: int) -> str:
    return (
        f"{index}. arxiv_id: {paper.arxiv_id}\n"
        f"title: {paper.title}\n"
        f"primary_category: {paper.primary_category}\n"
        f"categories: {', '.join(paper.categories)}\n"
        f"published: {paper.published.date().isoformat()}\n"
        f"comment: {paper.comment or 'N/A'}\n"
        f"summary: {truncate_summary(paper.summary, summary_max_chars)}"
    )


def truncate_summary(summary: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(summary) <= max_chars:
        return summary
    if max_chars <= 32:
        return summary[:max_chars].rstrip()
    tail_chars = min(1000, max_chars // 4)
    head_chars = max_chars - tail_chars - len("\n...[truncated]...\n")
    if head_chars <= 0:
        return summary[:max_chars].rstrip()
    return f"{summary[:head_chars].rstrip()}\n...[truncated]...\n{summary[-tail_chars:].lstrip()}"
