from __future__ import annotations

import json
from pathlib import Path

from arxiv_astro.models import PaperMetadata


def read_metadata(path: Path) -> list[PaperMetadata]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = [payload]
        return [PaperMetadata.model_validate(item) for item in payload]
    return read_metadata_jsonl(path)


def read_metadata_jsonl(path: Path) -> list[PaperMetadata]:
    papers: list[PaperMetadata] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            papers.append(PaperMetadata.model_validate_json(line))
    return papers
