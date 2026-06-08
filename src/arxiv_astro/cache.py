from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from arxiv_astro.models import LLMMetadata, PaperBlock, PaperContentBlock, PaperMetadata
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


def load_cached_interpretation(
    output_root: Path,
    paper: PaperMetadata,
    expected_metadata: LLMMetadata | None = None,
) -> PaperBlock | None:
    path = interpretation_path(output_root, paper)
    if not path.exists():
        return None
    try:
        block = PaperBlock.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError:
        return None
    if expected_metadata and not interpretation_metadata_matches(block.llm_metadata, expected_metadata):
        return None
    return block


def interpretation_metadata_matches(actual: LLMMetadata | None, expected: LLMMetadata) -> bool:
    if actual is None:
        return False
    return (
        actual.provider == expected.provider
        and actual.model == expected.model
        and actual.task == expected.task
        and actual.prompt_version == expected.prompt_version
        and actual.schema_version == expected.schema_version
        and actual.max_input_chars == expected.max_input_chars
    )
