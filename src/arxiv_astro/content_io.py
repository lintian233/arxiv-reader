from __future__ import annotations

import json
from pathlib import Path

from arxiv_astro.models import PaperContentBlock, RunManifest


def read_content_blocks(path: Path) -> list[PaperContentBlock]:
    if path.is_dir():
        path = path / "content.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if is_manifest_payload(payload):
        return [read_content_block_from_path(Path(output["content"])) for output in payload["outputs"] if output.get("content")]
    return [PaperContentBlock.model_validate(payload)]


def read_content_block_from_path(path: Path) -> PaperContentBlock:
    if path.is_dir():
        path = path / "content.json"
    return PaperContentBlock.model_validate_json(path.read_text(encoding="utf-8"))


def read_content_manifest(path: Path) -> RunManifest:
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def content_paths_from_manifest(path: Path) -> dict[str, Path]:
    manifest = read_content_manifest(path)
    return {output.arxiv_id: output.content for output in manifest.outputs if output.content}


def is_manifest_payload(payload: object) -> bool:
    return isinstance(payload, dict) and "outputs" in payload and "paper_ids" in payload
