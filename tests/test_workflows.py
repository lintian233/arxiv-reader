from __future__ import annotations

from arxiv_astro.models import ContentType, PaperContent, PaperContentBlock
from arxiv_astro.workflows import build_content_context, build_explain_context
from arxiv_astro.writer import write_content_outputs, write_fetch_outputs


def test_build_content_context_from_manifest(sample_paper, tmp_path) -> None:
    manifest_path = write_fetch_outputs([sample_paper], tmp_path, "astro-ph.IM", max_results=3, run_date="2024-01-01")

    context = build_content_context(manifest_path, [sample_paper])

    assert context.category == "astro-ph.IM"
    assert context.max_results == 3
    assert context.metadata_paths is not None
    assert context.metadata_paths[sample_paper.arxiv_id].name == "metadata.json"


def test_build_content_context_from_single_paper(sample_paper, tmp_path) -> None:
    input_path = tmp_path / "papers" / sample_paper.arxiv_id / "metadata.json"

    context = build_content_context(input_path, [sample_paper])

    assert context.category == sample_paper.primary_category
    assert context.max_results == 1
    assert context.metadata_paths is None


def test_build_explain_context_from_manifest(sample_paper, tmp_path) -> None:
    content_block = PaperContentBlock(
        paper=sample_paper,
        content=PaperContent(content_type=ContentType.HTML, text="Full text", text_chars=9),
    )
    manifest_path = write_content_outputs([content_block], tmp_path, "astro-ph.IM", max_results=2, run_date="2024-01-01")

    context = build_explain_context(manifest_path, [content_block], tmp_path)

    assert context.category == "astro-ph.IM"
    assert context.max_results == 2
    assert context.content_by_id[sample_paper.arxiv_id].content.text == "Full text"
    assert context.content_paths is not None
    assert context.content_paths[sample_paper.arxiv_id].name == "content.json"


def test_build_explain_context_from_single_content(sample_paper, tmp_path) -> None:
    content_block = PaperContentBlock(
        paper=sample_paper,
        content=PaperContent(content_type=ContentType.HTML, text="Full text", text_chars=9),
    )
    input_path = tmp_path / "papers" / sample_paper.arxiv_id / "content.json"

    context = build_explain_context(input_path, [content_block], tmp_path)

    assert context.category == sample_paper.primary_category
    assert context.max_results == 1
    assert context.metadata_paths is None
    assert context.content_paths is None
    assert context.figure_sets == {}
