from __future__ import annotations

import json
from pathlib import Path

from arxiv_astro.content_pipeline import load_content_blocks
from arxiv_astro.content_io import read_content_blocks
from arxiv_astro.explain_pipeline import build_image_context, build_llm_input, explain_content_blocks
from arxiv_astro.metadata_io import read_metadata
from arxiv_astro.models import ArticleImage, ContentType, LLMInterpretation, PaperContent, PaperContentBlock
from arxiv_astro.writer import (
    write_content_block,
    write_content_outputs,
    write_fetch_outputs,
    write_interpretation_outputs,
    write_metadata_block,
)


class FakeContentLoader:
    def load(self, paper):
        return PaperContent(
            content_type=ContentType.HTML,
            text=f"Full text for {paper.arxiv_id}",
            text_chars=28,
            source_url=paper.html_url,
            images=[ArticleImage(url="https://arxiv.org/html/2401.12345v1/fig1.png", caption="Figure caption")],
        )


class FakeLLMClient:
    def __init__(self) -> None:
        self.seen_text = ""

    def interpret(self, paper, text: str):
        self.seen_text = text
        return LLMInterpretation(
            one_sentence="一句话",
            background="背景",
            problem="问题",
            method="方法",
            result="结果",
            importance="重要性",
            limitations="限制",
        )


def test_read_metadata_from_manifest(sample_paper, tmp_path: Path) -> None:
    metadata_path = write_fetch_outputs([sample_paper], tmp_path, "astro-ph.IM", max_results=1, run_date="2024-01-01")

    papers = read_metadata(metadata_path)

    assert len(papers) == 1
    assert papers[0].arxiv_id == sample_paper.arxiv_id


def test_read_metadata_from_paper_json(sample_paper, tmp_path: Path) -> None:
    metadata_path = write_metadata_block(sample_paper, tmp_path, run_date="2024-01-01")

    papers = read_metadata(metadata_path)

    assert len(papers) == 1
    assert papers[0].entry_id == sample_paper.entry_id
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["fetched_date"] == "2024-01-01"


def test_load_content_blocks(sample_paper) -> None:
    blocks = load_content_blocks([sample_paper], FakeContentLoader())

    assert len(blocks) == 1
    assert blocks[0].paper.arxiv_id == sample_paper.arxiv_id
    assert blocks[0].content.content_type == ContentType.HTML
    assert blocks[0].content.text.startswith("Full text")


def test_write_content_block(sample_paper, tmp_path: Path) -> None:
    block = load_content_blocks([sample_paper], FakeContentLoader())[0]

    output_path = write_content_block(block, tmp_path, run_date="2024-01-01")

    assert output_path == tmp_path / "papers" / sample_paper.arxiv_id / "content.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["paper"]["arxiv_id"] == sample_paper.arxiv_id
    assert payload["content"]["text"].startswith("Full text")
    assert payload["loaded_date"] == "2024-01-01"


def test_write_content_outputs_manifest(sample_paper, tmp_path: Path) -> None:
    block = load_content_blocks([sample_paper], FakeContentLoader())[0]
    from arxiv_astro.models import FigureSet, LocalFigure

    figure_set = FigureSet(
        arxiv_id=sample_paper.arxiv_id,
        figures=[
            LocalFigure(
                index=1,
                url="https://arxiv.org/html/2401.12345v1/fig1.png",
                path=Path("figures/fig_001.png"),
                caption="Figure caption",
            )
        ],
    )

    output_path = write_content_outputs(
        [block],
        tmp_path,
        "astro-ph.IM",
        max_results=1,
        figure_sets={sample_paper.arxiv_id: figure_set},
        run_date="2024-01-01",
    )

    assert output_path == tmp_path / "runs" / "2024-01-01_astro-ph.IM" / "manifest.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["outputs"][0]["content"].endswith("content.json")
    assert payload["outputs"][0]["figures"].endswith("figures.json")
    figures_payload = json.loads(Path(payload["outputs"][0]["figures"]).read_text(encoding="utf-8"))
    assert figures_payload["figures"][0]["path"] == "figures/fig_001.png"


def test_read_content_blocks_from_manifest(sample_paper, tmp_path: Path) -> None:
    block = load_content_blocks([sample_paper], FakeContentLoader())[0]
    output_path = write_content_outputs([block], tmp_path, "astro-ph.IM", max_results=1)

    blocks = read_content_blocks(output_path)

    assert len(blocks) == 1
    assert blocks[0].content.text.startswith("Full text")


def test_read_content_blocks_from_paper_json(sample_paper, tmp_path: Path) -> None:
    block = load_content_blocks([sample_paper], FakeContentLoader())[0]
    output_path = write_content_block(block, tmp_path)

    blocks = read_content_blocks(output_path)

    assert len(blocks) == 1
    assert blocks[0].paper.arxiv_id == sample_paper.arxiv_id


def test_build_llm_input_includes_image_context(sample_paper) -> None:
    block = PaperContentBlock(
        paper=sample_paper,
        content=PaperContent(
            content_type=ContentType.HTML,
            text="Full text",
            text_chars=9,
            source_url=sample_paper.html_url,
            images=[
                ArticleImage(
                    url="https://arxiv.org/html/2401.12345v1/fig1.png",
                    alt="Figure alt",
                    caption="Figure caption",
                )
            ],
        ),
    )

    assert "图1:" in build_image_context(block)
    llm_input = build_llm_input(block)
    assert "Full text" in llm_input
    assert "Figure caption" in llm_input


def test_explain_content_blocks(sample_paper) -> None:
    llm = FakeLLMClient()
    content_block = PaperContentBlock(
        paper=sample_paper,
        content=PaperContent(
            content_type=ContentType.HTML,
            text="abcdef",
            text_chars=6,
            source_url=sample_paper.html_url,
        ),
    )

    blocks = explain_content_blocks([content_block], llm, max_input_chars=3)

    assert llm.seen_text == "abc"
    assert blocks[0].paper.arxiv_id == sample_paper.arxiv_id
    assert blocks[0].source.used_chars == 3
    assert blocks[0].source.image_count == 0
    assert blocks[0].source.source_url == sample_paper.html_url
    assert blocks[0].llm_interpretation.one_sentence == "一句话"


def test_write_interpretations(sample_paper, tmp_path: Path) -> None:
    content_block = PaperContentBlock(
        paper=sample_paper,
        content=PaperContent(content_type=ContentType.ABSTRACT, text="abstract", text_chars=8),
    )
    block = explain_content_blocks([content_block], FakeLLMClient(), max_input_chars=20)[0]

    from arxiv_astro.models import FigureSet

    figure_set = FigureSet(arxiv_id=sample_paper.arxiv_id)
    manifest_path = write_interpretation_outputs(
        [block],
        {sample_paper.arxiv_id: content_block},
        tmp_path,
        "astro-ph.IM",
        max_results=1,
        figure_sets={sample_paper.arxiv_id: figure_set},
        run_date="2024-01-01",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    interpretation_path = Path(payload["outputs"][0]["interpretation"])
    reader_path = Path(payload["outputs"][0]["reader"])
    assert json.loads(interpretation_path.read_text(encoding="utf-8"))["llm_interpretation"]["one_sentence"] == "一句话"
    reader_payload = json.loads(reader_path.read_text(encoding="utf-8"))
    assert reader_payload["content"]["text"] == "abstract"
    assert reader_payload["figures"]["figures"] == []
