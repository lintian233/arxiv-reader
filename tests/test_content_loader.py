from __future__ import annotations

from pathlib import Path

import httpx

from arxiv_astro.content_loader import ContentLoader, build_content, extract_html_text, normalize_text
from arxiv_astro.models import ContentType


class FakePdfParser:
    def extract_text(self, pdf_path: Path) -> str:
        assert pdf_path.exists()
        return "PDF text"


def test_extract_html_text_removes_noise() -> None:
    html = "<html><body><nav>menu</nav><main><h1>Title</h1><script>x</script><p>Body text</p></main></body></html>"

    assert extract_html_text(html) == "Title\nBody text"


def test_loader_prefers_html(sample_paper, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/html/" in str(request.url):
            return httpx.Response(200, text="<main><p>HTML paper text</p></main>")
        raise AssertionError("PDF should not be fetched when HTML succeeds")

    loader = ContentLoader(
        pdf_dir=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        pdf_parser=FakePdfParser(),
    )

    content = loader.load(sample_paper)

    assert content.content_type == ContentType.HTML
    assert content.text == "HTML paper text"


def test_loader_uses_pdf_when_html_missing(sample_paper, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/html/" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(200, content=b"%PDF")

    loader = ContentLoader(
        pdf_dir=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        pdf_parser=FakePdfParser(),
    )

    content = loader.load(sample_paper)

    assert content.content_type == ContentType.PDF
    assert content.text == "PDF text"


def test_loader_falls_back_to_abstract(sample_paper, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    loader = ContentLoader(
        pdf_dir=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        pdf_parser=FakePdfParser(),
    )

    content = loader.load(sample_paper)

    assert content.content_type == ContentType.ABSTRACT
    assert content.text == sample_paper.summary


def test_build_content_and_normalize_text() -> None:
    content = build_content(ContentType.ABSTRACT, " A  line\n\n second\tline ")

    assert normalize_text(" A  line\n\n second\tline ") == "A line\nsecond line"
    assert content.text_chars == len("A line\nsecond line")


def test_loader_ignores_invalid_proxy_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NO_PROXY", "[ff00::*]")
    monkeypatch.setenv("no_proxy", "[ff00::*]")

    loader = ContentLoader(pdf_dir=tmp_path)

    assert isinstance(loader, ContentLoader)
