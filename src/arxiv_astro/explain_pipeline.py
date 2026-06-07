from __future__ import annotations

from arxiv_astro.llm_client import LLMClient
from arxiv_astro.models import PaperBlock, PaperContentBlock
from arxiv_astro.normalize import build_paper_block, truncate_for_llm
from arxiv_astro.settings import debug_log


def explain_content_blocks(
    content_blocks: list[PaperContentBlock],
    llm_client: LLMClient,
    max_input_chars: int,
) -> list[PaperBlock]:
    blocks: list[PaperBlock] = []
    for content_block in content_blocks:
        paper = content_block.paper
        llm_input = truncate_for_llm(build_llm_input(content_block), max_input_chars)
        debug_log(
            "explaining paper",
            arxiv_id=paper.arxiv_id,
            content_type=content_block.content.content_type,
            input_chars=len(llm_input),
        )
        interpretation = llm_client.interpret(paper, llm_input)
        blocks.append(build_paper_block(paper, content_block.content, interpretation, llm_input))
        debug_log("explained paper", arxiv_id=paper.arxiv_id)
    return blocks


def build_llm_input(content_block: PaperContentBlock) -> str:
    content = content_block.content
    image_context = build_image_context(content_block)
    if image_context:
        return f"{content.text}\n\n图片信息:\n{image_context}"
    return content.text


def build_image_context(content_block: PaperContentBlock) -> str:
    lines: list[str] = []
    for index, image in enumerate(content_block.content.images, start=1):
        parts = [f"图{index}: {image.url}"]
        if image.alt:
            parts.append(f"alt={image.alt}")
        if image.caption:
            parts.append(f"caption={image.caption}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)
