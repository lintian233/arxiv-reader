from __future__ import annotations

from arxiv_astro.arxiv_client import ArxivClient
from arxiv_astro.content_loader import ContentLoader
from arxiv_astro.explain_pipeline import build_llm_input
from arxiv_astro.llm_client import LLMClient
from arxiv_astro.models import PaperBlock, PaperContent, PaperContentBlock, PaperMetadata
from arxiv_astro.normalize import build_paper_block, truncate_for_llm


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

    def run(self, category: str, max_results: int) -> list[PaperBlock]:
        blocks: list[PaperBlock] = []
        for paper in self.arxiv_client.fetch_category(category, max_results=max_results):
            content = self.content_loader.load(paper)
            used_text = truncate_for_llm(build_llm_input_for_paper(paper, content), self.max_input_chars)
            interpretation = self.llm_client.interpret(paper, used_text)
            blocks.append(build_paper_block(paper, content, interpretation, used_text))
        return blocks


def build_llm_input_for_paper(paper: PaperMetadata, content: PaperContent) -> str:
    return build_llm_input(PaperContentBlock(paper=paper, content=content))
