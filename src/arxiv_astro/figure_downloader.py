from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import httpx

from arxiv_astro.http_client import create_http_client
from arxiv_astro.models import ArticleImage, FigureSet, LocalFigure, PaperMetadata
from arxiv_astro.settings import debug_log
from arxiv_astro.writer import paper_file


class FigureDownloader:
    def __init__(
        self,
        output_root: Path,
        http_client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.output_root = output_root
        self._client = http_client or create_http_client(timeout=timeout, follow_redirects=True)

    def download(self, paper: PaperMetadata, images: list[ArticleImage]) -> FigureSet:
        figures: list[LocalFigure] = []
        for index, image in enumerate(images, start=1):
            local_path = figure_relative_path(index, str(image.url))
            target_path = paper_file(self.output_root, paper.arxiv_id, str(local_path))
            if target_path.exists():
                figures.append(local_figure(index, image, local_path))
                continue
            if self.download_image(str(image.url), target_path):
                figures.append(local_figure(index, image, local_path))
        return FigureSet(arxiv_id=paper.arxiv_id, figures=figures)

    def download_image(self, url: str, target_path: Path) -> bool:
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError:
            debug_log("figure download failed", url=url)
            return False
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(response.content)
        return True


def local_figure(index: int, image: ArticleImage, local_path: Path) -> LocalFigure:
    return LocalFigure(index=index, url=image.url, path=local_path, alt=image.alt, caption=image.caption)


def figure_relative_path(index: int, url: str) -> Path:
    return Path("figures") / f"fig_{index:03d}{image_extension(url)}"


def image_extension(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}:
        return suffix
    return ".img"
