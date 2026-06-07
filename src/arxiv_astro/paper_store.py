from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from arxiv_astro.models import (
    FigureSet,
    MetadataBlock,
    PaperBlock,
    PaperContentBlock,
    PaperStatus,
    ReaderPaperBlock,
)
from arxiv_astro.writer import paper_file, safe_arxiv_id


class PaperStore:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def paper_dir(self, arxiv_id: str) -> Path:
        return self.output_root / "papers" / safe_arxiv_id(arxiv_id)

    def list_paper_dirs(self) -> list[Path]:
        papers_dir = self.output_root / "papers"
        if not papers_dir.exists():
            return []
        return sorted(path for path in papers_dir.iterdir() if path.is_dir())

    def list_arxiv_ids(self) -> list[str]:
        return [path.name for path in self.list_paper_dirs()]

    def list_statuses(self) -> list[PaperStatus]:
        return [self.status(path.name) for path in self.list_paper_dirs()]

    def status(self, arxiv_id: str) -> PaperStatus:
        metadata = self.load_metadata(arxiv_id)
        reader = self.load_reader(arxiv_id)
        paper = metadata.paper if metadata else reader.paper if reader else None
        return PaperStatus(
            arxiv_id=paper.arxiv_id if paper else arxiv_id,
            title=paper.title if paper else None,
            primary_category=paper.primary_category if paper else None,
            has_metadata=self.exists(arxiv_id, "metadata.json"),
            has_content=self.exists(arxiv_id, "content.json"),
            has_figures=self.exists(arxiv_id, "figures.json"),
            has_interpretation=self.exists(arxiv_id, "interpretation.json"),
            has_reader=self.exists(arxiv_id, "reader.json"),
            has_pdf=self.exists(arxiv_id, "paper.pdf"),
        )

    def load_metadata(self, arxiv_id: str) -> MetadataBlock | None:
        return load_model(paper_file(self.output_root, arxiv_id, "metadata.json"), MetadataBlock)

    def load_content(self, arxiv_id: str) -> PaperContentBlock | None:
        return load_model(paper_file(self.output_root, arxiv_id, "content.json"), PaperContentBlock)

    def load_figures(self, arxiv_id: str) -> FigureSet | None:
        return load_model(paper_file(self.output_root, arxiv_id, "figures.json"), FigureSet)

    def load_interpretation(self, arxiv_id: str) -> PaperBlock | None:
        return load_model(paper_file(self.output_root, arxiv_id, "interpretation.json"), PaperBlock)

    def load_reader(self, arxiv_id: str) -> ReaderPaperBlock | None:
        return load_model(paper_file(self.output_root, arxiv_id, "reader.json"), ReaderPaperBlock)

    def exists(self, arxiv_id: str, filename: str) -> bool:
        return paper_file(self.output_root, arxiv_id, filename).exists()


def load_model(path: Path, model_class):
    if not path.exists():
        return None
    try:
        return model_class.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        return None
