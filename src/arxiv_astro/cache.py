from __future__ import annotations

from pathlib import Path

from arxiv_astro.models import PaperBlock, PaperContentBlock, PaperMetadata
from arxiv_astro.writer import paper_file


def content_path(output_root: Path, paper: PaperMetadata) -> Path:
    return paper_file(output_root, paper.arxiv_id, "content.json")


def interpretation_path(output_root: Path, paper: PaperMetadata) -> Path:
    return paper_file(output_root, paper.arxiv_id, "interpretation.json")


def load_cached_content(output_root: Path, paper: PaperMetadata) -> PaperContentBlock | None:
    path = content_path(output_root, paper)
    if not path.exists():
        return None
    return PaperContentBlock.model_validate_json(path.read_text(encoding="utf-8"))


def load_cached_interpretation(output_root: Path, paper: PaperMetadata) -> PaperBlock | None:
    path = interpretation_path(output_root, paper)
    if not path.exists():
        return None
    return PaperBlock.model_validate_json(path.read_text(encoding="utf-8"))
