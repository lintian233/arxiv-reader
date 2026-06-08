from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from arxiv_astro.models import (
    FigureSet,
    MetadataBlock,
    PaperBlock,
    PaperContentBlock,
    PaperMetadata,
    ReaderPaperBlock,
    RunManifest,
    RunOutput,
)
from arxiv_astro.normalize import build_reader_block


def current_date() -> str:
    return date.today().isoformat()


def write_metadata_block(paper: PaperMetadata, output_root: Path, run_date: str | None = None) -> Path:
    return write_json(
        paper_file(output_root, paper.arxiv_id, "metadata.json"),
        MetadataBlock(paper=paper, fetched_date=run_date or current_date()),
    )


def write_content_block(block: PaperContentBlock, output_root: Path, run_date: str | None = None) -> Path:
    path = paper_file(output_root, block.paper.arxiv_id, "content.json")
    if path.exists():
        return path
    block.loaded_date = run_date or current_date()
    return write_json(path, block)


def write_figure_set(figure_set: FigureSet, output_root: Path, run_date: str | None = None) -> Path:
    path = paper_file(output_root, figure_set.arxiv_id, "figures.json")
    if path.exists():
        return path
    figure_set.downloaded_date = run_date or current_date()
    return write_json(path, figure_set)


def write_interpretation_block(block: PaperBlock, output_root: Path, run_date: str | None = None) -> Path:
    path = paper_file(output_root, block.paper.arxiv_id, "interpretation.json")
    if path.exists():
        return path
    block.interpreted_date = run_date or current_date()
    return write_json(path, block)


def write_reader_block(block: ReaderPaperBlock, output_root: Path, run_date: str | None = None) -> Path:
    block.built_date = run_date or current_date()
    return write_json(paper_file(output_root, block.paper.arxiv_id, "reader.json"), block)


def write_fetch_outputs(
    papers: list[PaperMetadata],
    output_root: Path,
    category: str,
    max_results: int,
    run_date: str | None = None,
) -> Path:
    effective_date = run_date or current_date()
    outputs = [
        RunOutput(arxiv_id=paper.arxiv_id, metadata=write_metadata_block(paper, output_root, effective_date))
        for paper in papers
    ]
    return write_manifest(output_root, category, max_results, effective_date, outputs)


def write_content_outputs(
    blocks: list[PaperContentBlock],
    output_root: Path,
    category: str,
    max_results: int,
    metadata_paths: dict[str, Path] | None = None,
    figure_sets: dict[str, FigureSet] | None = None,
    run_date: str | None = None,
) -> Path:
    effective_date = run_date or current_date()
    outputs = []
    for block in blocks:
        content_path = write_content_block(block, output_root, effective_date)
        figures_path = write_figure_set(figure_sets[block.paper.arxiv_id], output_root, effective_date) if figure_sets else None
        outputs.append(
            RunOutput(
                arxiv_id=block.paper.arxiv_id,
                metadata=metadata_paths.get(block.paper.arxiv_id, default_metadata_path(output_root, block.paper.arxiv_id))
                if metadata_paths
                else default_metadata_path(output_root, block.paper.arxiv_id),
                content=content_path,
                figures=figures_path,
            )
        )
    return write_manifest(output_root, category, max_results, effective_date, outputs)


def write_interpretation_outputs(
    blocks: list[PaperBlock],
    contents: dict[str, PaperContentBlock],
    output_root: Path,
    category: str,
    max_results: int,
    metadata_paths: dict[str, Path] | None = None,
    content_paths: dict[str, Path] | None = None,
    figure_sets: dict[str, FigureSet] | None = None,
    figure_paths: dict[str, Path] | None = None,
    run_date: str | None = None,
) -> Path:
    effective_date = run_date or current_date()
    outputs = []
    for block in blocks:
        content_block = contents[block.paper.arxiv_id]
        figure_set = figure_sets.get(block.paper.arxiv_id) if figure_sets else None
        interpretation_path = write_interpretation_block(block, output_root, effective_date)
        if figure_paths and block.paper.arxiv_id in figure_paths:
            figures_path = figure_paths[block.paper.arxiv_id]
        elif figure_set:
            figures_path = write_figure_set(figure_set, output_root, effective_date)
        else:
            figures_path = None
        reader_path = write_reader_block(build_reader_block(content_block.content, block, figure_set), output_root, effective_date)
        outputs.append(
            RunOutput(
                arxiv_id=block.paper.arxiv_id,
                metadata=metadata_paths.get(block.paper.arxiv_id, default_metadata_path(output_root, block.paper.arxiv_id))
                if metadata_paths
                else default_metadata_path(output_root, block.paper.arxiv_id),
                content=content_paths.get(block.paper.arxiv_id, default_content_path(output_root, block.paper.arxiv_id))
                if content_paths
                else default_content_path(output_root, block.paper.arxiv_id),
                figures=figures_path,
                interpretation=interpretation_path,
                reader=reader_path,
            )
        )
    return write_manifest(output_root, category, max_results, effective_date, outputs)


def write_reader_outputs(
    blocks: list[ReaderPaperBlock],
    output_root: Path,
    category: str,
    max_results: int,
    run_date: str | None = None,
) -> Path:
    effective_date = run_date or current_date()
    outputs = []
    for block in blocks:
        metadata_path = write_metadata_block(block.paper, output_root, effective_date)
        content_path = write_content_block(
            PaperContentBlock(paper=block.paper, content=block.content, loaded_date=effective_date),
            output_root,
            effective_date,
        )
        figures_path = write_figure_set(block.figures, output_root, effective_date) if block.figures else None
        interpretation_path = write_interpretation_block(
            PaperBlock(
                paper=block.paper,
                source=block.source,
                llm_interpretation=block.llm_interpretation,
                llm_metadata=block.llm_metadata,
                interpreted_date=effective_date,
            ),
            output_root,
            effective_date,
        )
        reader_path = write_reader_block(block, output_root, effective_date)
        outputs.append(
            RunOutput(
                arxiv_id=block.paper.arxiv_id,
                metadata=metadata_path,
                content=content_path,
                figures=figures_path,
                interpretation=interpretation_path,
                reader=reader_path,
            )
        )
    return write_manifest(output_root, category, max_results, effective_date, outputs)


def write_manifest(
    output_root: Path,
    category: str,
    max_results: int,
    run_date: str,
    outputs: list[RunOutput],
) -> Path:
    manifest = RunManifest(
        run_id=run_id(run_date, category),
        category=category,
        max_results=max_results,
        run_date=run_date,
        paper_ids=[output.arxiv_id for output in outputs],
        outputs=outputs,
    )
    return write_json(output_root / "runs" / manifest.run_id / "manifest.json", manifest)


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        data = payload.model_dump(mode="json")
    else:
        data = payload
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def paper_file(output_root: Path, arxiv_id: str, filename: str) -> Path:
    return output_root / "papers" / safe_arxiv_id(arxiv_id) / filename


def default_metadata_path(output_root: Path, arxiv_id: str) -> Path:
    return paper_file(output_root, arxiv_id, "metadata.json")


def default_content_path(output_root: Path, arxiv_id: str) -> Path:
    return paper_file(output_root, arxiv_id, "content.json")


def default_figures_path(output_root: Path, arxiv_id: str) -> Path:
    return paper_file(output_root, arxiv_id, "figures.json")


def run_id(run_date: str, category: str) -> str:
    return f"{run_date}_{safe_category(category)}"


def safe_category(category: str) -> str:
    return category.replace("/", "_")


def safe_arxiv_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")
