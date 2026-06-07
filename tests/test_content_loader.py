from __future__ import annotations

from pathlib import Path

import httpx

from arxiv_astro.content_loader import (
    ContentLoader,
    build_content,
    extract_html_images,
    extract_html_text,
    first_srcset_url,
    normalize_text,
    resolve_image_url,
)
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
            return httpx.Response(
                200,
                text="""
                <main>
                  <p>HTML paper text</p>
                  <figure>
                    <img src="figures/fig1.png" alt=" First figure ">
                    <figcaption> Figure 1. Caption text. </figcaption>
                  </figure>
                </main>
                """,
            )
        raise AssertionError("PDF should not be fetched when HTML succeeds")

    loader = ContentLoader(
        pdf_dir=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        pdf_parser=FakePdfParser(),
    )

    content = loader.load(sample_paper)

    assert content.content_type == ContentType.HTML
    assert "HTML paper text" in content.text
    assert "Figure 1. Caption text." in content.text
    assert str(content.source_url) == str(sample_paper.html_url)
    assert len(content.images) == 1
    assert str(content.images[0].url) == "https://arxiv.org/html/figures/fig1.png"
    assert content.images[0].alt == "First figure"
    assert content.images[0].caption == "Figure 1. Caption text."


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
    assert str(content.source_url) == str(sample_paper.pdf_url)


def test_load_pdf_content_directly(sample_paper, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/pdf/" in str(request.url)
        return httpx.Response(200, content=b"%PDF")

    loader = ContentLoader(
        pdf_dir=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        pdf_parser=FakePdfParser(),
    )

    content = loader.load_pdf_content(sample_paper)

    assert content is not None
    assert content.content_type == ContentType.PDF
    assert content.text == "PDF text"
    assert str(content.source_url) == str(sample_paper.pdf_url)


def test_load_pdf_content_returns_none_on_download_failure(sample_paper, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    loader = ContentLoader(
        pdf_dir=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        pdf_parser=FakePdfParser(),
    )

    assert loader.load_pdf_content(sample_paper) is None


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
    assert str(content.source_url) == str(sample_paper.abs_url)


def test_build_content_and_normalize_text() -> None:
    content = build_content(ContentType.ABSTRACT, " A  line\n\n second\tline ")

    assert normalize_text(" A  line\n\n second\tline ") == "A line\nsecond line"
    assert content.text_chars == len("A line\nsecond line")


def test_extract_html_images_resolves_urls_and_deduplicates() -> None:
    html = """
    <body>
      <main>
        <figure>
          <img src="fig1.png" alt=" Alpha ">
          <figcaption> Caption A </figcaption>
        </figure>
        <img src="fig1.png" alt="Duplicate">
        <figure><img srcset="fig2-small.png 1x, fig2-large.png 2x"></figure>
        <img src="data:image/png;base64,abc">
      </main>
    </body>
    """

    images = extract_html_images(html, "https://arxiv.org/html/2401.12345v1")

    assert [str(image.url) for image in images] == [
        "https://arxiv.org/html/fig1.png",
        "https://arxiv.org/html/fig2-small.png",
    ]
    assert images[0].alt == "Alpha"
    assert images[0].caption == "Caption A"
    assert images[1].alt is None


def test_image_url_helpers() -> None:
    assert resolve_image_url("fig.png", "https://arxiv.org/html/1234") == "https://arxiv.org/html/fig.png"
    assert resolve_image_url("data:image/png;base64,abc", "https://arxiv.org/html/1234") is None
    assert first_srcset_url("small.png 1x, large.png 2x") == "small.png"


def test_loader_ignores_invalid_proxy_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NO_PROXY", "[ff00::*]")
    monkeypatch.setenv("no_proxy", "[ff00::*]")

    loader = ContentLoader(pdf_dir=tmp_path)

    assert isinstance(loader, ContentLoader)
