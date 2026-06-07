from __future__ import annotations

from arxiv_astro.content_loader import ContentLoader
from arxiv_astro.models import PaperContentBlock, PaperMetadata
from arxiv_astro.settings import debug_log


def load_content_blocks(papers: list[PaperMetadata], content_loader: ContentLoader) -> list[PaperContentBlock]:
    blocks: list[PaperContentBlock] = []
    for paper in papers:
        debug_log("loading paper content", arxiv_id=paper.arxiv_id)
        content = content_loader.load(paper)
        debug_log(
            "loaded paper content",
            arxiv_id=paper.arxiv_id,
            content_type=content.content_type,
            text_chars=content.text_chars,
            image_count=len(content.images),
        )
        blocks.append(PaperContentBlock(paper=paper, content=content))
    return blocks
