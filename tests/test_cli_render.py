from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from arxiv_astro.cli_render import (
    render_fetch_summary,
    render_next_steps,
    render_output_path,
    render_pipeline_summary,
    render_selection_summary,
    render_stage,
)
from arxiv_astro.models import (
    ContentType,
    LLMInterpretation,
    LLMMetadata,
    PaperContent,
    PaperSelectionSummary,
    ReaderPaperBlock,
    SelectionBlock,
    SelectedPaper,
    SourceUsage,
)


def capture_console() -> tuple[Console, StringIO]:
    output = StringIO()
    return Console(file=output, width=140, force_terminal=False), output


def test_render_fetch_summary_shows_metadata(sample_paper) -> None:
    console, output = capture_console()

    render_fetch_summary([sample_paper], "astro-ph.IM", 1, Path("data/runs/run/manifest.json"), console)

    text = output.getvalue()
    assert "arxiv-astro fetch: astro-ph.IM (1/1)" in text
    assert sample_paper.arxiv_id in text
    assert sample_paper.primary_category in text
    assert sample_paper.title in text
    assert "saved: data/runs/run/manifest.json" in text


def test_render_selection_summary_shows_selected_papers(sample_paper) -> None:
    console, output = capture_console()
    selection = SelectionBlock(
        category="astro-ph",
        fetch_results=100,
        max_results=1,
        interests="FRB",
        candidate_ids=[sample_paper.arxiv_id],
        selected=[
            SelectedPaper(
                arxiv_id=sample_paper.arxiv_id,
                relevance=5,
                matched_interests=["FRB", "transient"],
                reason="matches the configured interests",
            )
        ],
        summary=PaperSelectionSummary(
            candidate_count=100,
            requested_count=3,
            selected_count=1,
            shortfall=2,
            shortfall_reason="候选中只有一篇高度相关论文。",
        ),
        llm_metadata=LLMMetadata(
            provider="openai-compatible",
            model="model",
            task="paper_selection",
            prompt_version="v2",
            schema_version="v1",
            max_input_chars=10000,
        ),
    )

    render_selection_summary(selection, [sample_paper], console)

    text = output.getvalue()
    assert "LLM selection: 1/3" in text
    assert "candidates: 100  requested: 3  selected: 1  shortfall: 2" in text
    assert "shortfall reason: 候选中只有一篇高度相关论文。" in text
    assert sample_paper.arxiv_id in text
    assert "FRB, transient" in text
    assert sample_paper.title in text
    assert "matches the configured interests" in text


def test_render_output_path() -> None:
    console, output = capture_console()

    render_output_path("saved", Path("data/runs/run/manifest.json"), console)

    assert "saved: data/runs/run/manifest.json" in output.getvalue()


def test_render_stage_and_next_steps() -> None:
    console, output = capture_console()

    render_stage("1/4", "Fetch candidates", "category: astro-ph  candidates: 60", console)
    render_next_steps(["arxiv-astro report --input data/runs/run/manifest.json"], console)

    text = output.getvalue()
    assert "[1/4] Fetch candidates" in text
    assert "category: astro-ph  candidates: 60" in text
    assert "Next:" in text
    assert "arxiv-astro report --input data/runs/run/manifest.json" in text


def test_render_pipeline_summary_shows_all_completed_blocks(sample_paper) -> None:
    console, output = capture_console()
    blocks = [
        ReaderPaperBlock(
            paper=sample_paper.model_copy(update={"arxiv_id": f"2401.1234{index}v1", "title": f"Paper {index} " + "x" * 80}),
            content=PaperContent(content_type=ContentType.HTML, text="full text", text_chars=100 + index),
            source=SourceUsage(content_type=ContentType.HTML, text_chars=100 + index, used_chars=80),
            llm_interpretation=LLMInterpretation(
                one_sentence=f"summary {index} " + "y" * 120,
                problem_context="problem",
                why_it_matters="matters",
                what_the_paper_does="does",
                main_results="results",
                key_figures=[
                    {"index": 1, "plain_caption": "关键图说明", "why_key": "支撑核心结论", "evidence": None}
                ]
                if index == 1
                else [],
                limitations="limits",
                field_position="field",
            ),
        )
        for index in range(1, 4)
    ]

    render_pipeline_summary(blocks, console)

    text = output.getvalue()
    assert "2401.12341v1" in text
    assert "2401.12342v1" in text
    assert "2401.12343v1" in text
    assert "Completed papers: 3/3" not in text
    assert "1/3 2401.12341v1" in text
    assert "Summary:" in text
    assert "summary 1 " + "y" * 120 in text
    assert "Core result:" in text
    assert "results" in text
    assert "Key figures:" in text
    assert "图1: 关键图说明" in text
