from __future__ import annotations

import json
from pathlib import Path

from arxiv_astro.models import (
    ArticleImage,
    ContentType,
    FigureSet,
    LLMInterpretation,
    LocalFigure,
    PaperContent,
)
from arxiv_astro.normalize import build_paper_block, build_reader_block
from arxiv_astro.report import build_report, generate_report, site_url
from arxiv_astro.writer import write_reader_block, write_reader_outputs


def interpretation_with_key_figure() -> LLMInterpretation:
    return LLMInterpretation(
        one_sentence="一句话总结",
        problem_context="问题背景",
        why_it_matters="为什么重要",
        what_the_paper_does="做了什么",
        main_results="核心结果",
        key_figures=[
            {
                "index": 1,
                "plain_caption": "这张图展示核心趋势",
                "why_key": "它直接支撑核心结果",
                "evidence": "趋势随时间增强",
            }
        ],
        limitations="限制",
        field_position="领域位置",
    )


def reader_block_with_figure(sample_paper):
    content = PaperContent(
        content_type=ContentType.HTML,
        text="Full text",
        text_chars=9,
        source_url=sample_paper.html_url,
        images=[
            ArticleImage(
                url="https://arxiv.org/html/2401.12345v1/fig1.png",
                alt="figure alt",
                caption="Original caption",
            )
        ],
    )
    paper_block = build_paper_block(sample_paper, content, interpretation_with_key_figure(), "Full text")
    figures = FigureSet(
        arxiv_id=sample_paper.arxiv_id,
        figures=[
            LocalFigure(
                index=1,
                url="https://arxiv.org/html/2401.12345v1/fig1.png",
                path=Path("figures/fig_001.png"),
                alt="local alt",
                caption="Original caption",
            )
        ],
    )
    return build_reader_block(content, paper_block, figures)


def test_generate_report_from_manifest_renders_key_figure(sample_paper, tmp_path: Path) -> None:
    reader = reader_block_with_figure(sample_paper)
    figure_path = tmp_path / "papers" / sample_paper.arxiv_id / "figures" / "fig_001.png"
    figure_path.parent.mkdir(parents=True)
    figure_path.write_bytes(b"image")
    manifest_path = write_reader_outputs([reader], tmp_path, "astro-ph.IM", max_results=1, run_date="2024-01-01")

    output_path = generate_report(manifest_path, tmp_path)

    assert output_path == tmp_path / "runs" / "2024-01-01_astro-ph.IM" / "report.html"
    html = output_path.read_text(encoding="utf-8")
    assert "astro-ph.IM paper briefing" in html
    assert sample_paper.title in html
    assert "这张图展示核心趋势" in html
    assert "/papers/2401.12345v1/figures/fig_001.png" in html


def test_generate_report_from_reader_json(sample_paper, sample_interpretation, tmp_path: Path) -> None:
    content = PaperContent(content_type=ContentType.ABSTRACT, text="abstract", text_chars=8)
    reader = build_reader_block(content, build_paper_block(sample_paper, content, sample_interpretation, "abstract"))
    reader_path = write_reader_block(reader, tmp_path, run_date="2024-01-01")

    output_path = generate_report(reader_path, tmp_path)

    assert output_path == tmp_path / "papers" / sample_paper.arxiv_id / "report.html"
    html = output_path.read_text(encoding="utf-8")
    assert "No key figures were selected" in html
    assert "领域位置" in html


def test_build_report_rejects_manifest_without_reader(sample_paper, tmp_path: Path) -> None:
    manifest = {
        "run_id": "2024-01-01_astro-ph.IM",
        "category": "astro-ph.IM",
        "max_results": 1,
        "run_date": "2024-01-01",
        "paper_ids": [sample_paper.arxiv_id],
        "outputs": [{"arxiv_id": sample_paper.arxiv_id, "metadata": str(tmp_path / "metadata.json")}],
    }
    manifest_path = tmp_path / "runs" / "2024-01-01_astro-ph.IM" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        build_report(manifest_path, tmp_path)
    except ValueError as exc:
        assert "has no reader path" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_site_url_uses_output_root(tmp_path: Path) -> None:
    target = tmp_path / "papers" / "paper" / "figures" / "fig.png"

    assert site_url(target, tmp_path) == "/papers/paper/figures/fig.png"
