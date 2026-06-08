from __future__ import annotations

from datetime import datetime, timezone

import pytest

from arxiv_astro.models import LLMInterpretation, PaperMetadata


@pytest.fixture
def sample_paper() -> PaperMetadata:
    return PaperMetadata(
        entry_id="http://arxiv.org/abs/2401.12345v1",
        arxiv_id="2401.12345v1",
        title="A Test Paper",
        authors=["Ada Lovelace", "Grace Hopper"],
        summary="This paper tests a pipeline.",
        published=datetime(2024, 1, 1, tzinfo=timezone.utc),
        updated=datetime(2024, 1, 2, tzinfo=timezone.utc),
        primary_category="astro-ph.CO",
        categories=["astro-ph.CO"],
        abs_url="https://arxiv.org/abs/2401.12345v1",
        pdf_url="https://arxiv.org/pdf/2401.12345v1",
        html_url="https://arxiv.org/html/2401.12345v1",
        doi="10.1234/example",
        journal_ref=None,
        comment="A test comment",
    )


@pytest.fixture
def sample_interpretation() -> LLMInterpretation:
    return LLMInterpretation(
        one_sentence="一句话总结",
        problem_context="问题背景",
        why_it_matters="为什么重要",
        what_the_paper_does="做了什么",
        main_results="核心结果",
        key_figures=[],
        limitations="限制",
        field_position="领域位置",
    )
