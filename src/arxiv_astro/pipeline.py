from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from arxiv_astro.arxiv_client import ArxivClient
from arxiv_astro.content_loader import ContentLoader
from arxiv_astro.explain_pipeline import build_llm_input
from arxiv_astro.llm_client import LLMClient
from arxiv_astro.models import PaperBlock, PaperContent, PaperContentBlock, PaperMetadata, ReaderPaperBlock
from arxiv_astro.normalize import build_paper_block, build_reader_block, truncate_for_llm


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
    ) -> None:
        self.arxiv_client = arxiv_client
        self.content_loader = content_loader
        self.llm_client = llm_client
        self.max_input_chars = max_input_chars

    def run(self, category: str, max_results: int, on_update: PipelineUpdate | None = None) -> list[ReaderPaperBlock]:
        papers = self.arxiv_client.fetch_category(category, max_results=max_results)
        emit_pipeline_started(on_update, papers)
        return [
            self.process_paper(PaperRun(paper=paper, index=index, total=len(papers)), on_update)
            for index, paper in enumerate(papers, start=1)
        ]

    def process_paper(self, run: PaperRun, on_update: PipelineUpdate | None = None) -> ReaderPaperBlock:
        emit_paper_update(on_update, run, status="fetched")

        content = self.load_content(run, on_update)
        used_text = self.prepare_llm_input(run.paper, content)
        emit_paper_update(on_update, run, status="llm_started", content=content, used_chars=len(used_text))

        interpretation = self.llm_client.interpret(run.paper, used_text)
        block = build_paper_block(run.paper, content, interpretation, used_text)
        reader_block = build_reader_block(content, block)
        emit_paper_update(on_update, run, status="done", content=content, used_chars=len(used_text), block=block)
        return reader_block

    def load_content(self, run: PaperRun, on_update: PipelineUpdate | None = None) -> PaperContent:
        content = self.content_loader.load(run.paper)
        emit_paper_update(on_update, run, status="content_loaded", content=content)
        return content

    def prepare_llm_input(self, paper: PaperMetadata, content: PaperContent) -> str:
        return truncate_for_llm(build_llm_input_for_paper(paper, content), self.max_input_chars)


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
