from __future__ import annotations

from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from arxiv_astro.http_client import create_http_client
from arxiv_astro.models import ContentType, PaperContent, PaperMetadata
from arxiv_astro.pdf_parser import PdfParser


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
        html_text = self._load_html(str(paper.html_url))
        if html_text:
            return build_content(ContentType.HTML, html_text)

        pdf_text = self._load_pdf(paper)
        if pdf_text:
            return build_content(ContentType.PDF, pdf_text)

        return build_content(ContentType.ABSTRACT, paper.summary)

    def _load_html(self, html_url: str) -> str:
        try:
            response = self._client.get(html_url)
            if response.status_code == 404:
                return ""
            response.raise_for_status()
        except httpx.HTTPError:
            return ""
        return extract_html_text(response.text)

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


def build_content(content_type: ContentType, text: str) -> PaperContent:
    cleaned = normalize_text(text)
    return PaperContent(content_type=content_type, text=cleaned, text_chars=len(cleaned))


def extract_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for node in soup(["script", "style", "nav", "footer"]):
        node.decompose()
    main = soup.find("main") or soup.body or soup
    return normalize_text(main.get_text("\n"))


def normalize_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
