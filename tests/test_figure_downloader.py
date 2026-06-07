from __future__ import annotations

import httpx

from arxiv_astro.figure_downloader import FigureDownloader, figure_relative_path, image_extension
from arxiv_astro.models import ArticleImage


def test_figure_downloader_downloads_images_with_relative_paths(sample_paper, tmp_path) -> None:
    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, content=b"PNG")

    downloader = FigureDownloader(
        output_root=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    figure_set = downloader.download(
        sample_paper,
        [
            ArticleImage(
                url="https://arxiv.org/html/2401.12345v1/fig1.png",
                alt="Figure alt",
                caption="Figure caption",
            )
        ],
    )

    assert figure_set.arxiv_id == sample_paper.arxiv_id
    assert figure_set.figures[0].path.as_posix() == "figures/fig_001.png"
    assert figure_set.figures[0].caption == "Figure caption"
    assert (tmp_path / "papers" / sample_paper.arxiv_id / "figures" / "fig_001.png").read_bytes() == b"PNG"
    assert seen_urls == ["https://arxiv.org/html/2401.12345v1/fig1.png"]


def test_figure_downloader_uses_existing_file_cache(sample_paper, tmp_path) -> None:
    target = tmp_path / "papers" / sample_paper.arxiv_id / "figures" / "fig_001.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"cached")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("image should not be downloaded when target exists")

    downloader = FigureDownloader(
        output_root=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    figure_set = downloader.download(
        sample_paper,
        [ArticleImage(url="https://arxiv.org/html/2401.12345v1/fig1.jpg")],
    )

    assert figure_set.figures[0].path.as_posix() == "figures/fig_001.jpg"
    assert target.read_bytes() == b"cached"


def test_figure_downloader_skips_failed_download(sample_paper, tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    downloader = FigureDownloader(
        output_root=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    figure_set = downloader.download(
        sample_paper,
        [ArticleImage(url="https://arxiv.org/html/2401.12345v1/fig1.png")],
    )

    assert figure_set.figures == []


def test_figure_path_helpers() -> None:
    assert figure_relative_path(2, "https://example.com/a/b/c.webp").as_posix() == "figures/fig_002.webp"
    assert image_extension("https://example.com/image?format=png") == ".img"
