from __future__ import annotations

from pathlib import Path

import fitz


class PdfParser:
    def extract_text(self, pdf_path: Path) -> str:
        with fitz.open(pdf_path) as document:
            pages = [page.get_text("text") for page in document]
        return "\n".join(page.strip() for page in pages if page.strip())
