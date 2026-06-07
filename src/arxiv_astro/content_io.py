from __future__ import annotations

import json
from pathlib import Path

from arxiv_astro.models import PaperContentBlock


def read_content_blocks(path: Path) -> list[PaperContentBlock]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = [payload]
        return [PaperContentBlock.model_validate(item) for item in payload]
    return read_content_blocks_jsonl(path)


def read_content_blocks_jsonl(path: Path) -> list[PaperContentBlock]:
    blocks: list[PaperContentBlock] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            blocks.append(PaperContentBlock.model_validate_json(line))
    return blocks
