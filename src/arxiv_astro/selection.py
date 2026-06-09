from __future__ import annotations

from dataclasses import dataclass

from arxiv_astro.llm_client import LLMClient
from arxiv_astro.llm_tasks import PaperSelectionTask
from arxiv_astro.models import PaperMetadata, SelectionBlock, SelectedPaper


class SelectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SelectionResult:
    papers: list[PaperMetadata]
    block: SelectionBlock


class PaperSelector:
    def __init__(
        self,
        llm_client: LLMClient,
        max_input_chars: int,
        summary_max_chars: int,
        task: PaperSelectionTask | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.max_input_chars = max_input_chars
        self.summary_max_chars = summary_max_chars
        self.task = task or PaperSelectionTask()

    def select(
        self,
        papers: list[PaperMetadata],
        interests: str,
        max_results: int,
        category: str,
        fetch_results: int,
    ) -> SelectionResult:
        if not interests.strip():
            raise SelectionError("paper interests are empty")
        try:
            task_result = self.task.run(
                self.llm_client,
                papers,
                interests,
                max_results,
                self.max_input_chars,
                self.summary_max_chars,
            )
        except Exception as exc:
            raise SelectionError(str(exc)) from exc

        selected = normalize_selected(task_result.value.selected, papers, max_results)
        selected_ids = [item.arxiv_id for item in selected]
        by_id = {paper.arxiv_id: paper for paper in papers}
        selected_papers = [by_id[arxiv_id] for arxiv_id in selected_ids]
        block = SelectionBlock(
            category=category,
            fetch_results=fetch_results,
            max_results=max_results,
            interests=interests,
            candidate_ids=[paper.arxiv_id for paper in papers],
            selected=selected,
            llm_metadata=task_result.metadata,
        )
        return SelectionResult(papers=selected_papers, block=block)


def normalize_selected(
    selected: list[SelectedPaper],
    candidates: list[PaperMetadata],
    max_results: int,
) -> list[SelectedPaper]:
    candidate_ids = {paper.arxiv_id for paper in candidates}
    normalized: list[SelectedPaper] = []
    seen: set[str] = set()
    for item in selected:
        if item.arxiv_id not in candidate_ids or item.arxiv_id in seen:
            continue
        normalized.append(item)
        seen.add(item.arxiv_id)
        if len(normalized) >= max_results:
            break
    return normalized
