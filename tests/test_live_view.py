from __future__ import annotations

from io import StringIO

from rich.console import Console

from arxiv_astro.live_view import PipelineLiveRenderer
from arxiv_astro.models import ArticleImage, ContentType, LLMInterpretation, PaperBlock, PaperContent, SourceUsage


def test_pipeline_live_renderer_tracks_updates(sample_paper) -> None:
    renderer = PipelineLiveRenderer(console=Console(file=StringIO(), force_terminal=False))

    renderer.on_update({"event": "fetched", "total": 1})
    renderer.on_update({"event": "ignored"})
    renderer.on_update({"event": "paper", "index": 1, "total": 1, "paper": sample_paper, "status": "fetched"})
    renderer.on_update(
        {
            "event": "paper",
            "index": 1,
            "total": 1,
            "paper": sample_paper,
            "status": "content_loaded",
            "content": PaperContent(
                content_type=ContentType.HTML,
                text="full text",
                text_chars=9,
                source_url=sample_paper.html_url,
                images=[
                    ArticleImage(
                        url="https://arxiv.org/html/2401.12345v1/fig1.png",
                        caption="caption",
                    )
                ],
            ),
        }
    )

    row = renderer.rows[sample_paper.arxiv_id]
    assert renderer.total == 1
    assert row.status == "content_loaded"
    assert row.source == "html"
    assert row.text_chars == 9
    assert row.image_count == 1


def test_pipeline_live_renderer_tracks_llm_result(sample_paper) -> None:
    renderer = PipelineLiveRenderer(console=Console(file=StringIO(), force_terminal=False))
    block = PaperBlock(
        paper=sample_paper,
        source=SourceUsage(content_type=ContentType.HTML, text_chars=9, used_chars=9),
        llm_interpretation=LLMInterpretation(
            one_sentence="这是一句话总结",
            background="背景",
            problem="问题",
            method="方法",
            result="主要结果",
            importance="重要性",
            limitations="限制",
        ),
    )

    renderer.on_update(
        {
            "event": "paper",
            "index": 1,
            "total": 1,
            "paper": sample_paper,
            "status": "done",
            "content": PaperContent(content_type=ContentType.HTML, text="full text", text_chars=9),
            "block": block,
        }
    )

    row = renderer.rows[sample_paper.arxiv_id]
    assert row.one_sentence == "这是一句话总结"
    assert row.result == "主要结果"

    output = StringIO()
    Console(file=output, force_terminal=False, width=220).print(renderer.render())
    rendered = output.getvalue()
    assert "LLM interpretations" in rendered
    assert "这是一句话总结" in rendered
    assert "主要结果" in rendered


def test_pipeline_live_renderer_context_manager(sample_paper) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False)

    with PipelineLiveRenderer(console=console) as renderer:
        renderer.on_update({"event": "fetched", "total": 1})
        renderer.on_update({"event": "paper", "index": 1, "total": 1, "paper": sample_paper, "status": "done"})

    rendered = output.getvalue()
    assert "arxiv-astro pipeline" in rendered
    assert sample_paper.arxiv_id in rendered
