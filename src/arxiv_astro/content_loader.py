from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from pydantic import ValidationError

from arxiv_astro.http_client import create_http_client
from arxiv_astro.models import ArticleImage, ContentType, PaperContent, PaperMetadata
from arxiv_astro.pdf_parser import PdfParser
from arxiv_astro.settings import debug_log


class ContentLoader:
    def __init__(
        self,
        pdf_dir: Path,
        http_client: httpx.Client | None = None,
        pdf_parser: PdfParser | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.pdf_dir = pdf_dir
        self._client = http_client or create_http_client(timeout=timeout, follow_redirects=True)
        self._pdf_parser = pdf_parser or PdfParser()

    def load(self, paper: PaperMetadata) -> PaperContent:
        html_content = self._load_html(str(paper.html_url))
        if html_content:
            return html_content

        pdf_content = self.load_pdf_content(paper)
        if pdf_content:
            return pdf_content

        return build_content(ContentType.ABSTRACT, paper.summary, source_url=str(paper.abs_url))

    def load_pdf_content(self, paper: PaperMetadata) -> PaperContent | None:
        pdf_text = self._load_pdf(paper)
        if not pdf_text:
            return None
        return build_content(ContentType.PDF, pdf_text, source_url=str(paper.pdf_url))

    def _load_html(self, html_url: str) -> PaperContent | None:
        try:
            response = self._client.get(html_url)
            if response.status_code == 404:
                debug_log("arxiv html not found", url=html_url)
                return None
            response.raise_for_status()
        except httpx.HTTPError:
            debug_log("arxiv html load failed", url=html_url)
            return None
        text = extract_html_text(response.text)
        if not text:
            return None
        images = extract_html_images(response.text, html_url)
        debug_log("loaded arxiv html", url=html_url, text_chars=len(text), image_count=len(images))
        return build_content(ContentType.HTML, text, source_url=html_url, images=images)

    def _load_pdf(self, paper: PaperMetadata) -> str:
        try:
            response = self._client.get(str(paper.pdf_url))
            response.raise_for_status()
        except httpx.HTTPError:
            return ""

        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = self.pdf_dir / f"{paper.arxiv_id.replace('/', '_')}.pdf"
        pdf_path.write_bytes(response.content)
        return self._pdf_parser.extract_text(pdf_path)


def build_content(
    content_type: ContentType,
    text: str,
    source_url: str | None = None,
    images: list[ArticleImage] | None = None,
) -> PaperContent:
    cleaned = normalize_text(text)
    return PaperContent(
        content_type=content_type,
        text=cleaned,
        text_chars=len(cleaned),
        source_url=source_url,
        images=images or [],
    )


def extract_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for node in soup(["script", "style", "nav", "footer"]):
        node.decompose()
    main = soup.find("main") or soup.body or soup
    return normalize_text(main.get_text("\n"))


def extract_html_images(html: str, base_url: str) -> list[ArticleImage]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.body or soup
    images: list[ArticleImage] = []
    seen_urls: set[str] = set()
    for img in main.find_all("img"):
        if not img.find_parent("figure"):
            continue
        image_url = resolve_image_url(img.get("src") or img.get("data-src") or first_srcset_url(img.get("srcset")), base_url)
        if not image_url or image_url in seen_urls:
            continue
        seen_urls.add(image_url)
        image = build_article_image(
            url=image_url,
            alt=optional_normalize_text(img.get("alt")),
            caption=find_image_caption(img),
        )
        if image:
            images.append(image)
    return images


def resolve_image_url(raw_url: str | None, base_url: str) -> str | None:
    if not raw_url or raw_url.startswith("data:"):
        return None
    absolute_url = urljoin(base_url, raw_url)
    if not absolute_url.startswith(("http://", "https://")):
        return None
    return absolute_url


def first_srcset_url(srcset: str | None) -> str | None:
    if not srcset:
        return None
    first_candidate = srcset.split(",")[0].strip()
    return first_candidate.split()[0] if first_candidate else None


def find_image_caption(img) -> str | None:
    figure = img.find_parent("figure")
    caption = figure.find("figcaption")
    if not caption:
        return None
    return optional_normalize_text(caption.get_text(" "))


def build_article_image(url: str, alt: str | None, caption: str | None) -> ArticleImage | None:
    try:
        return ArticleImage(url=url, alt=alt, caption=caption)
    except ValidationError:
        return None


def normalize_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def optional_normalize_text(text: str | None) -> str | None:
    if not text:
        return None
    normalized = normalize_text(text)
    return normalized or None
