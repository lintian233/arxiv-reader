from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arxiv_astro.arxiv_client import ArxivClient
from arxiv_astro.cache import load_cached_content, load_cached_interpretation
from arxiv_astro.content_loader import ContentLoader
from arxiv_astro.explain_pipeline import build_llm_input
from arxiv_astro.figure_downloader import FigureDownloader
from arxiv_astro.llm_client import LLMClient
from arxiv_astro.llm_tasks import PaperInterpretationTask
from arxiv_astro.models import FigureSet, PaperBlock, PaperContent, PaperContentBlock, PaperMetadata, ReaderPaperBlock
from arxiv_astro.normalize import build_paper_block, build_reader_block, truncate_for_llm
from arxiv_astro.selection import PaperSelector, SelectionError


PipelineUpdate = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class PaperRun:
    paper: PaperMetadata
    index: int
    total: int


class Pipeline:
    def __init__(
        self,
        arxiv_client: ArxivClient,
        content_loader: ContentLoader,
        llm_client: LLMClient,
        max_input_chars: int,
        cache_root: Path | None = None,
        figure_downloader: FigureDownloader | None = None,
        interpretation_task: PaperInterpretationTask | None = None,
        paper_selector: PaperSelector | None = None,
    ) -> None:
        self.arxiv_client = arxiv_client
        self.content_loader = content_loader
        self.llm_client = llm_client
        self.max_input_chars = max_input_chars
        self.cache_root = cache_root
        self.figure_downloader = figure_downloader
        self.interpretation_task = interpretation_task or PaperInterpretationTask()
        self.paper_selector = paper_selector
        self.selection_block = None

    def run(
        self,
        category: str,
        max_results: int,
        on_update: PipelineUpdate | None = None,
        fetch_results: int | None = None,
        interests: str | None = None,
    ) -> list[ReaderPaperBlock]:
        self.selection_block = None
        papers = self.fetch_papers(category, max_results, fetch_results, interests)
        emit_pipeline_started(on_update, papers)
        return [
            self.process_paper(PaperRun(paper=paper, index=index, total=len(papers)), on_update)
            for index, paper in enumerate(papers, start=1)
        ]

    def fetch_papers(
        self,
        category: str,
        max_results: int,
        fetch_results: int | None = None,
        interests: str | None = None,
    ) -> list[PaperMetadata]:
        papers = self.fetch_candidates(category, max_results, fetch_results, interests)
        return self.select_papers(papers, category, max_results, fetch_results, interests)

    def fetch_candidates(
        self,
        category: str,
        max_results: int,
        fetch_results: int | None = None,
        interests: str | None = None,
    ) -> list[PaperMetadata]:
        self.selection_block = None
        if interests and fetch_results is None:
            return self.arxiv_client.fetch_latest_day(category)
        effective_fetch_results = fetch_results if interests else max_results
        return self.arxiv_client.fetch_category(category, max_results=effective_fetch_results or max_results)

    def select_papers(
        self,
        papers: list[PaperMetadata],
        category: str,
        max_results: int,
        fetch_results: int | None = None,
        interests: str | None = None,
    ) -> list[PaperMetadata]:
        if not interests:
            return papers
        if not self.paper_selector:
            raise SelectionError("paper selector is required when interests are set")
        effective_fetch_results = fetch_results if interests else max_results
        selection = self.paper_selector.select(
            papers,
            interests,
            max_results=max_results,
            category=category,
            fetch_results=effective_fetch_results or len(papers),
        )
        self.selection_block = selection.block
        return selection.papers

    def process_paper(self, run: PaperRun, on_update: PipelineUpdate | None = None) -> ReaderPaperBlock:
        emit_paper_update(on_update, run, status="fetched")

        content = self.load_content(run, on_update)
        figures = self.download_figures(run.paper, content)
        used_text = self.prepare_llm_input(run.paper, content)

        block = self.interpret_paper(run, content, used_text, on_update)
        reader_block = build_reader_block(content, block, figures)
        emit_paper_update(on_update, run, status="done", content=content, used_chars=block.source.used_chars, block=block)
        return reader_block

    def load_content(self, run: PaperRun, on_update: PipelineUpdate | None = None) -> PaperContent:
        if self.cache_root:
            cached = load_cached_content(self.cache_root, run.paper)
            if cached:
                emit_paper_update(on_update, run, status="content_loaded", content=cached.content, cache_hit=True)
                return cached.content

        content = self.content_loader.load(run.paper)
        emit_paper_update(on_update, run, status="content_loaded", content=content)
        return content

    def interpret_paper(
        self,
        run: PaperRun,
        content: PaperContent,
        used_text: str,
        on_update: PipelineUpdate | None = None,
    ) -> PaperBlock:
        expected_metadata = self.interpretation_task.metadata(self.llm_client, self.max_input_chars)
        cached = (
            load_cached_interpretation(self.cache_root, run.paper, expected_metadata=expected_metadata)
            if self.cache_root
            else None
        )
        if cached:
            emit_paper_update(
                on_update,
                run,
                status="llm_started",
                content=content,
                used_chars=cached.source.used_chars,
                cache_hit=True,
            )
            return cached

        emit_paper_update(on_update, run, status="llm_started", content=content, used_chars=len(used_text))
        result = self.interpretation_task.run(self.llm_client, run.paper, used_text, self.max_input_chars)
        return build_paper_block(run.paper, content, result.value, used_text, result.metadata)

    def prepare_llm_input(self, paper: PaperMetadata, content: PaperContent) -> str:
        return truncate_for_llm(build_llm_input_for_paper(paper, content), self.max_input_chars)

    def download_figures(self, paper: PaperMetadata, content: PaperContent) -> FigureSet | None:
        if not self.figure_downloader:
            return None
        return self.figure_downloader.download(paper, content.images)


def emit_update(on_update: PipelineUpdate | None, payload: dict[str, Any]) -> None:
    if on_update:
        on_update(payload)


def emit_pipeline_started(on_update: PipelineUpdate | None, papers: list[PaperMetadata]) -> None:
    emit_update(on_update, {"event": "fetched", "total": len(papers)})


def emit_paper_update(on_update: PipelineUpdate | None, run: PaperRun, status: str, **extra: Any) -> None:
    payload = {
        "event": "paper",
        "index": run.index,
        "total": run.total,
        "paper": run.paper,
        "status": status,
    }
    payload.update(extra)
    emit_update(on_update, payload)


def build_llm_input_for_paper(paper: PaperMetadata, content: PaperContent) -> str:
    return build_llm_input(PaperContentBlock(paper=paper, content=content))
