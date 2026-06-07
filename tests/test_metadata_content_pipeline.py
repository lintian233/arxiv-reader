from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from arxiv_astro.content_pipeline import load_content_blocks
from arxiv_astro.metadata_io import read_metadata
from arxiv_astro.models import ContentType, PaperContent
from arxiv_astro.writer import write_content_json, write_content_jsonl, write_metadata_json, write_metadata_jsonl


class FakeContentLoader:
    def load(self, paper):
        return PaperContent(
            content_type=ContentType.HTML,
            text=f"Full text for {paper.arxiv_id}",
            text_chars=28,
            source_url=paper.html_url,
        )


def test_read_metadata_jsonl(sample_paper, tmp_path: Path) -> None:
    metadata_path = write_metadata_jsonl([sample_paper], tmp_path, "astro-ph.IM")

    papers = read_metadata(metadata_path)

    assert len(papers) == 1
    assert papers[0].arxiv_id == sample_paper.arxiv_id


def test_read_metadata_json(sample_paper, tmp_path: Path) -> None:
    metadata_path = write_metadata_json([sample_paper], tmp_path, "astro-ph.IM")

    papers = read_metadata(metadata_path)

    assert len(papers) == 1
    assert papers[0].entry_id == sample_paper.entry_id


def test_load_content_blocks(sample_paper) -> None:
    blocks = load_content_blocks([sample_paper], FakeContentLoader())

    assert len(blocks) == 1
    assert blocks[0].paper.arxiv_id == sample_paper.arxiv_id
    assert blocks[0].content.content_type == ContentType.HTML
    assert blocks[0].content.text.startswith("Full text")


def test_write_content_jsonl(sample_paper, tmp_path: Path) -> None:
    block = load_content_blocks([sample_paper], FakeContentLoader())[0]

    output_path = write_content_jsonl([block], tmp_path, "metadata", now=datetime(2024, 1, 1, 1, 2, 3))

    assert output_path.name == "2024-01-01_010203_metadata_content.jsonl"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["paper"]["arxiv_id"] == sample_paper.arxiv_id
    assert payload["content"]["text"].startswith("Full text")


def test_write_content_json(sample_paper, tmp_path: Path) -> None:
    block = load_content_blocks([sample_paper], FakeContentLoader())[0]

    output_path = write_content_json([block], tmp_path, "metadata", now=datetime(2024, 1, 1, 1, 2, 3))

    assert output_path.name == "2024-01-01_010203_metadata_content.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["paper"]["entry_id"] == sample_paper.entry_id
