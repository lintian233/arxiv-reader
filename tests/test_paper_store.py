from __future__ import annotations

from pathlib import Path

from arxiv_astro.models import ContentType, PaperContent, PaperContentBlock
from arxiv_astro.normalize import build_paper_block, build_reader_block
from arxiv_astro.paper_store import PaperStore
from arxiv_astro.writer import (
    write_content_block,
    write_fetch_outputs,
    write_interpretation_block,
    write_reader_block,
)


def test_paper_store_empty_library(tmp_path: Path) -> None:
    store = PaperStore(tmp_path)

    assert store.list_paper_dirs() == []
    assert store.list_arxiv_ids() == []
    assert store.load_reader("2401.12345v1") is None


def test_paper_store_status_and_loaders(sample_paper, sample_interpretation, tmp_path: Path) -> None:
    content = PaperContent(content_type=ContentType.HTML, text="Full text", text_chars=9, source_url=sample_paper.html_url)
    interpretation = build_paper_block(sample_paper, content, sample_interpretation, "Full")
    reader = build_reader_block(content, interpretation)

    write_fetch_outputs([sample_paper], tmp_path, "astro-ph.IM", max_results=1, run_date="2024-01-01")
    write_content_block(PaperContentBlock(paper=sample_paper, content=content), tmp_path, run_date="2024-01-01")
    write_interpretation_block(interpretation, tmp_path, run_date="2024-01-01")
    write_reader_block(reader, tmp_path, run_date="2024-01-01")
    pdf_path = tmp_path / "papers" / sample_paper.arxiv_id / "paper.pdf"
    pdf_path.write_bytes(b"%PDF")

    store = PaperStore(tmp_path)
    status = store.status(sample_paper.arxiv_id)

    assert store.list_arxiv_ids() == [sample_paper.arxiv_id]
    assert status.arxiv_id == sample_paper.arxiv_id
    assert status.title == sample_paper.title
    assert status.primary_category == sample_paper.primary_category
    assert status.has_metadata is True
    assert status.has_content is True
    assert status.has_interpretation is True
    assert status.has_reader is True
    assert status.has_pdf is True
    assert store.load_metadata(sample_paper.arxiv_id).paper.title == sample_paper.title
    assert store.load_content(sample_paper.arxiv_id).content.text == "Full text"
    assert store.load_interpretation(sample_paper.arxiv_id).llm_interpretation.main_results == "核心结果"
    assert store.load_reader(sample_paper.arxiv_id).llm_interpretation.one_sentence == "一句话总结"


def test_paper_store_lists_statuses(sample_paper, tmp_path: Path) -> None:
    write_fetch_outputs([sample_paper], tmp_path, "astro-ph.IM", max_results=1)

    statuses = PaperStore(tmp_path).list_statuses()

    assert len(statuses) == 1
    assert statuses[0].has_metadata is True
    assert statuses[0].has_content is False


def test_paper_store_ignores_invalid_json(sample_paper, tmp_path: Path) -> None:
    paper_dir = tmp_path / "papers" / sample_paper.arxiv_id
    paper_dir.mkdir(parents=True)
    (paper_dir / "metadata.json").write_text("{not valid", encoding="utf-8")

    store = PaperStore(tmp_path)

    assert store.load_metadata(sample_paper.arxiv_id) is None
    status = store.status(sample_paper.arxiv_id)
    assert status.has_metadata is True
    assert status.title is None
