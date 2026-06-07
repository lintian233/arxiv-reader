from __future__ import annotations

import json
from pathlib import Path

from arxiv_astro.models import MetadataBlock, PaperMetadata, RunManifest


def read_metadata(path: Path) -> list[PaperMetadata]:
    if path.is_dir():
        path = path / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if is_manifest_payload(payload):
        return [read_metadata_from_path(Path(output["metadata"])) for output in payload["outputs"]]
    return [metadata_from_payload(payload)]


def read_metadata_from_path(path: Path) -> PaperMetadata:
    if path.is_dir():
        path = path / "metadata.json"
    return metadata_from_payload(json.loads(path.read_text(encoding="utf-8")))


def metadata_from_payload(payload: dict) -> PaperMetadata:
    if "paper" in payload:
        return MetadataBlock.model_validate(payload).paper
    return PaperMetadata.model_validate(payload)


def read_metadata_manifest(path: Path) -> RunManifest:
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def metadata_paths_from_manifest(path: Path) -> dict[str, Path]:
    manifest = read_metadata_manifest(path)
    return {output.arxiv_id: output.metadata for output in manifest.outputs}


def manifest_context(path: Path) -> tuple[str, int]:
    if path.name != "manifest.json":
        return ("papers", 1)
    manifest = read_metadata_manifest(path)
    return (manifest.category, manifest.max_results)


def is_manifest_payload(payload: object) -> bool:
    return isinstance(payload, dict) and "outputs" in payload and "paper_ids" in payload
