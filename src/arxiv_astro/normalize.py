from __future__ import annotations

from arxiv_astro.models import LLMInterpretation, PaperBlock, PaperContent, PaperMetadata, ReaderPaperBlock, SourceUsage


def truncate_for_llm(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def build_paper_block(
    paper: PaperMetadata,
    content: PaperContent,
    interpretation: LLMInterpretation,
    used_text: str,
) -> PaperBlock:
    return PaperBlock(
        paper=paper,
        source=SourceUsage(
            content_type=content.content_type,
            text_chars=content.text_chars,
            used_chars=len(used_text),
            source_url=content.source_url,
            image_count=len(content.images),
        ),
        llm_interpretation=interpretation,
    )


def build_reader_block(content: PaperContent, interpretation_block: PaperBlock) -> ReaderPaperBlock:
    return ReaderPaperBlock(
        paper=interpretation_block.paper,
        content=content,
        source=interpretation_block.source,
        llm_interpretation=interpretation_block.llm_interpretation,
    )
