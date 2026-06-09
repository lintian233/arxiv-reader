from __future__ import annotations

import json
from typing import Any

from arxiv_astro.llm_client import LLMClient
from arxiv_astro.llm_tasks.base import LLMTask, LLMTaskResult
from arxiv_astro.models import PaperMetadata, PaperSelectionResult


DEFAULT_SUMMARY_MAX_CHARS = 4000


class PaperSelectionTask(LLMTask[PaperSelectionResult]):
    task_name = "paper_selection"
    prompt_version = "v2"
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
    return """
你是一个面向天文学研究阅读的 arXiv 论文筛选助手。你的任务不是关键词匹配，而是根据候选论文 metadata 判断哪些论文最值得进入后续全文解读。

输出约束：
- 只返回合法 JSON，不要 Markdown，不要添加额外字段。
- 只能选择候选列表中真实存在的 arxiv_id，不能编造论文。
- 最多返回用户要求的 max_results 篇论文。
- selected 必须按 relevance 从高到低排序。
- matched_interests 写实际匹配到的用户兴趣词或短语。
- reason 用一句话说明具体匹配了哪些维度，例如科学问题、研究对象、仪器/数据、方法、结果价值或领域阅读价值。

请从以下维度判断相关性和阅读价值：
1. Scientific / technical problem：论文研究的问题是否直接触及用户兴趣，还是只是同属宽泛领域。
2. Object / phenomenon / data / instrument：研究对象、物理现象、观测数据、仪器、巡天或样本是否与用户兴趣相关。
3. Method / pipeline / modeling：方法、数据处理、建模、校准、分类、统计推断或系统误差分析是否与用户兴趣相关。
4. Contribution clarity：摘要是否显示清楚贡献、结果或改进，而不是只有背景描述。
5. Field-reading value：即使不是完全命中关键词，是否有助于理解该方向的前沿问题、主流方法、常见数据产品或技术瓶颈。

relevance 评分标准：
5 = must-read。科学/技术问题直接匹配用户兴趣，且对象/方法/贡献至少两个维度强相关，应优先进入全文解读。
4 = strong match。明显相关，对理解该兴趣方向有实际帮助，建议进入全文解读。
3 = useful adjacent。中等相关，可能是相邻问题、方法、数据、仪器或背景相关；当 4/5 不足时可以选择。
2 = weak match。只是关键词、宽泛领域或背景相关，不值得进入后续全文解读。
1 = irrelevant。基本无关。

选择策略：
- 优先选择 relevance 5 和 4。
- 如果 5/4 不足 max_results，可以补充 relevance 3。
- 不要选择 relevance 1 或 2 来凑数。
- 在不选择明显无关论文的前提下，请尽量返回接近 max_results 的论文。
- 如果 relevance >= 3 的论文仍少于 max_results，可以少选，并在 shortfall_reason 中用一句话说明原因。
- 如果选满，shortfall_reason 使用空字符串。

反关键词陷阱：
- 标题或摘要出现用户关键词不代表一定相关。
- 如果摘要不能支持实际科学问题、方法、对象或结果相关，不能给 4/5。
- 没有直接关键词但问题、方法、对象或仪器高度相关，也可以选择。

必须严格使用以下 JSON 骨架：
{"selected":[{"arxiv_id":"","relevance":5,"matched_interests":[""],"reason":""}],"shortfall_reason":""}
""".strip()


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
        "请根据 system rubric 选择最值得进入后续全文解读的论文。"
        "请优先选 relevance 5 和 4 的论文；如果数量不足，可以选择 relevance 3 但仍有助于理解领域前沿或主流方法的论文。"
        "不要用 relevance 1 或 2 的论文凑数。"
        "reason 必须说明具体匹配原因，例如科学问题、研究对象、仪器/数据、方法、结果价值或领域阅读价值。"
        "如果 selected 数量少于 max_results，shortfall_reason 必须说明为什么候选中没有足够多 relevance >= 3 的论文。"
        "请只返回 JSON。"
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
