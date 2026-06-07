from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from arxiv_astro.cache import load_cached_content, load_cached_interpretation
from arxiv_astro.content_io import content_paths_from_manifest
from arxiv_astro.content_pipeline import load_content_blocks
from arxiv_astro.content_loader import ContentLoader
from arxiv_astro.explain_pipeline import explain_content_blocks
from arxiv_astro.figure_downloader import FigureDownloader
from arxiv_astro.llm_client import LLMClient
from arxiv_astro.metadata_io import manifest_context, metadata_paths_from_manifest
from arxiv_astro.models import FigureSet, PaperBlock, PaperContentBlock, PaperMetadata
from arxiv_astro.settings import debug_log
from arxiv_astro.writer import paper_file


@dataclass(frozen=True)
class ContentContext:
    category: str
    max_results: int
    metadata_paths: dict[str, Path] | None


@dataclass(frozen=True)
class ExplainContext:
    category: str
    max_results: int
    content_by_id: dict[str, PaperContentBlock]
    metadata_paths: dict[str, Path] | None
    content_paths: dict[str, Path] | None
    figure_sets: dict[str, FigureSet]
    figure_paths: dict[str, Path]


def build_content_context(input_path: Path, papers: list[PaperMetadata]) -> ContentContext:
    if input_path.name == "manifest.json":
        category, max_results = manifest_context(input_path)
        metadata_paths = metadata_paths_from_manifest(input_path)
    else:
        category, max_results = papers[0].primary_category, len(papers)
        metadata_paths = None
    return ContentContext(category=category, max_results=max_results, metadata_paths=metadata_paths)


def build_explain_context(
    input_path: Path,
    content_blocks: list[PaperContentBlock],
    output_root: Path,
) -> ExplainContext:
    if input_path.name == "manifest.json":
        category, max_results = manifest_context(input_path)
        content_paths = content_paths_from_manifest(input_path)
        metadata_paths = metadata_paths_from_manifest(input_path)
    else:
        category, max_results = content_blocks[0].paper.primary_category, len(content_blocks)
        content_paths = None
        metadata_paths = None
    content_by_id = {block.paper.arxiv_id: block for block in content_blocks}
    figure_sets = load_figure_sets(output_root, content_blocks)
    figure_paths = local_figure_paths(output_root, content_blocks)
    return ExplainContext(
        category=category,
        max_results=max_results,
        content_by_id=content_by_id,
        metadata_paths=metadata_paths,
        content_paths=content_paths,
        figure_sets=figure_sets,
        figure_paths=figure_paths,
    )


def load_content_blocks_with_cache(
    papers: list[PaperMetadata],
    loader: ContentLoader,
    output_root: Path,
) -> list[PaperContentBlock]:
    misses = []
    blocks = []
    for paper in papers:
        cached = load_cached_content(output_root, paper)
        if cached:
            debug_log("content cache hit", arxiv_id=paper.arxiv_id)
            blocks.append(cached)
        else:
            debug_log("content cache miss", arxiv_id=paper.arxiv_id)
            misses.append(paper)
    blocks.extend(load_content_blocks(misses, loader))
    return blocks


def explain_content_blocks_with_cache(
    content_blocks: list[PaperContentBlock],
    llm_client: LLMClient,
    max_input_chars: int,
    output_root: Path,
) -> list[PaperBlock]:
    misses = []
    blocks = []
    for content_block in content_blocks:
        cached = load_cached_interpretation(output_root, content_block.paper)
        if cached:
            debug_log("interpretation cache hit", arxiv_id=content_block.paper.arxiv_id)
            blocks.append(cached)
        else:
            debug_log("interpretation cache miss", arxiv_id=content_block.paper.arxiv_id)
            misses.append(content_block)
    blocks.extend(explain_content_blocks(misses, llm_client, max_input_chars))
    return blocks


def download_figure_sets(
    content_blocks: list[PaperContentBlock],
    figure_downloader: FigureDownloader,
) -> dict[str, FigureSet]:
    return {
        block.paper.arxiv_id: figure_downloader.download(block.paper, block.content.images)
        for block in content_blocks
    }


def load_figure_sets(output_root: Path, content_blocks: list[PaperContentBlock]) -> dict[str, FigureSet]:
    figure_sets = {}
    for block in content_blocks:
        path = paper_file(output_root, block.paper.arxiv_id, "figures.json")
        if path.exists():
            figure_sets[block.paper.arxiv_id] = FigureSet.model_validate_json(path.read_text(encoding="utf-8"))
    return figure_sets


def local_figure_paths(output_root: Path, content_blocks: list[PaperContentBlock]) -> dict[str, Path]:
    return {
        block.paper.arxiv_id: paper_file(output_root, block.paper.arxiv_id, "figures.json")
        for block in content_blocks
        if paper_file(output_root, block.paper.arxiv_id, "figures.json").exists()
    }
