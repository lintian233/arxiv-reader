from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from arxiv_astro.models import PaperBlock, PaperMetadata


def write_jsonl(blocks: list[PaperBlock], output_dir: Path, category: str, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    safe_category = category.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}_{safe_category}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for block in blocks:
            handle.write(block.model_dump_json() + "\n")
    return output_path


def write_metadata_jsonl(
    papers: list[PaperMetadata],
    output_dir: Path,
    category: str,
    now: datetime | None = None,
) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    safe_category = category.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}_{safe_category}_metadata.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(paper.model_dump_json() + "\n")
    return output_path


def write_metadata_json(
    papers: list[PaperMetadata],
    output_dir: Path,
    category: str,
    now: datetime | None = None,
) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    safe_category = category.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}_{safe_category}_metadata.json"
    payload = [paper.model_dump(mode="json") for paper in papers]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path
