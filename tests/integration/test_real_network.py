from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from arxiv_astro.arxiv_client import ArxivClient
from arxiv_astro.content_loader import ContentLoader
from arxiv_astro.figure_downloader import FigureDownloader
from arxiv_astro.models import ContentType, PaperMetadata


pytestmark = pytest.mark.integration


def real_network_enabled() -> bool:
    return os.getenv("RUN_REAL_NETWORK", "").strip().lower() in {"1", "true", "yes", "on"}


def integration_category() -> str:
    return os.getenv("ARXIV_ASTRO_TEST_CATEGORY", "astro-ph.IM")


def image_test_arxiv_id() -> str:
    return os.getenv("ARXIV_ASTRO_IMAGE_TEST_ID", "2606.06234v1")


@pytest.fixture(scope="module")
def real_paper() -> PaperMetadata:
    if not real_network_enabled():
        pytest.skip("set RUN_REAL_NETWORK=1 to run real arXiv network tests")

    papers = ArxivClient(page_size=1, num_retries=1).fetch_category(integration_category(), max_results=1)
    assert papers, "arXiv returned no papers"
    return papers[0]


@pytest.fixture(scope="module")
def image_rich_paper() -> PaperMetadata:
    if not real_network_enabled():
        pytest.skip("set RUN_REAL_NETWORK=1 to run real arXiv network tests")

    arxiv_id = image_test_arxiv_id()
    return PaperMetadata(
        entry_id=f"http://arxiv.org/abs/{arxiv_id}",
        arxiv_id=arxiv_id,
        title="Known arXiv HTML paper with figures",
        authors=["integration-test"],
        summary="Known paper used to verify full text and article image extraction.",
        published=datetime(2026, 6, 4, tzinfo=timezone.utc),
        updated=datetime(2026, 6, 4, tzinfo=timezone.utc),
        primary_category=integration_category(),
        categories=[integration_category()],
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        html_url=f"https://arxiv.org/html/{arxiv_id}",
    )


def test_real_arxiv_fetch_returns_metadata(real_paper: PaperMetadata) -> None:
    assert real_paper.entry_id.startswith("http")
    assert real_paper.arxiv_id
    assert real_paper.title
    assert real_paper.authors
    assert real_paper.summary
    assert real_paper.primary_category
    assert real_paper.categories
    assert str(real_paper.abs_url).startswith("https://arxiv.org/abs/")
    assert str(real_paper.pdf_url).startswith("https://arxiv.org/pdf/")
    assert str(real_paper.html_url).startswith("https://arxiv.org/html/")


def test_real_content_loader_returns_full_text(real_paper: PaperMetadata, tmp_path: Path) -> None:
    content = ContentLoader(output_root=tmp_path).load(real_paper)

    assert content.content_type in {ContentType.HTML, ContentType.PDF, ContentType.ABSTRACT}
    assert content.source_url is not None
    assert content.text
    assert content.text_chars == len(content.text)
    assert content.text_chars >= len(real_paper.summary)
    if content.content_type == ContentType.HTML:
        assert str(content.source_url) == str(real_paper.html_url)


def test_real_content_loader_returns_full_text_and_article_images(
    image_rich_paper: PaperMetadata,
    tmp_path: Path,
) -> None:
    content = ContentLoader(output_root=tmp_path).load(image_rich_paper)

    assert content.content_type == ContentType.HTML
    assert str(content.source_url) == str(image_rich_paper.html_url)
    assert content.text_chars > 10_000
    assert len(content.images) > 0

    image_urls = [str(image.url) for image in content.images]
    assert all(url.startswith(str(image_rich_paper.html_url).rstrip("/") + "/") for url in image_urls)
    assert all("/static/" not in url for url in image_urls)
    assert any(image.alt or image.caption for image in content.images)


def test_real_content_loader_returns_pdf_text_for_same_arxiv_id(
    image_rich_paper: PaperMetadata,
    tmp_path: Path,
) -> None:
    content = ContentLoader(output_root=tmp_path).load_pdf_content(image_rich_paper)

    assert content is not None
    assert content.content_type == ContentType.PDF
    assert str(content.source_url) == str(image_rich_paper.pdf_url)
    assert content.text_chars > 10_000
    assert (tmp_path / "papers" / image_rich_paper.arxiv_id.replace("/", "_") / "paper.pdf").exists()


def test_real_figure_downloader_downloads_first_html_figure(
    image_rich_paper: PaperMetadata,
    tmp_path: Path,
) -> None:
    content = ContentLoader(output_root=tmp_path).load(image_rich_paper)

    figure_set = FigureDownloader(output_root=tmp_path).download(image_rich_paper, content.images[:1])

    assert len(figure_set.figures) == 1
    figure_path = tmp_path / "papers" / image_rich_paper.arxiv_id.replace("/", "_") / figure_set.figures[0].path
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
