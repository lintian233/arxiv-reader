from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from arxiv_astro.metadata_io import read_metadata_manifest
from arxiv_astro.models import KeyFigureInsight, LocalFigure, ReaderPaperBlock, RunManifest
from arxiv_astro.writer import paper_file, run_id, run_manifest_path, safe_arxiv_id


@dataclass(frozen=True)
class ReportContext:
    title: str
    category: str
    run_date: str
    paper_count: int
    papers: list[dict[str, Any]]


def generate_report(input_path: Path, paper_root: Path, runs_root: Path | None = None) -> Path:
    context, output_path = build_report(input_path, paper_root, runs_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(context), encoding="utf-8")
    return output_path


def build_report(input_path: Path, paper_root: Path, runs_root: Path | None = None) -> tuple[ReportContext, Path]:
    input_path = input_path.resolve()
    paper_root = paper_root.resolve()
    effective_runs_root = runs_root.resolve() if runs_root else paper_root
    if input_path.name == "manifest.json":
        manifest = read_metadata_manifest(input_path)
        blocks = read_reader_blocks_from_manifest(manifest)
        output_path = input_path.parent / "report.html"
        return build_report_context(blocks, output_path, paper_root, manifest), output_path

    block = read_reader_block(input_path)
    output_path = run_manifest_path(
        effective_runs_root,
        run_id(block.built_date or "manual", f"{block.paper.primary_category}_{block.paper.arxiv_id}"),
    ).parent / "report.html"
    return build_report_context([block], output_path, paper_root), output_path


def read_reader_blocks_from_manifest(manifest: RunManifest) -> list[ReaderPaperBlock]:
    blocks = []
    for output in manifest.outputs:
        if output.reader is None:
            raise ValueError(f"manifest output for {output.arxiv_id} has no reader path")
        blocks.append(read_reader_block(output.reader))
    return blocks


def read_reader_block(path: Path) -> ReaderPaperBlock:
    if path.is_dir():
        path = path / "reader.json"
    return ReaderPaperBlock.model_validate_json(path.read_text(encoding="utf-8"))


def build_report_context(
    blocks: list[ReaderPaperBlock],
    report_path: Path,
    paper_root: Path,
    manifest: RunManifest | None = None,
) -> ReportContext:
    category = manifest.category if manifest else infer_category(blocks)
    run_date = manifest.run_date if manifest else infer_run_date(blocks)
    return ReportContext(
        title=f"arxiv-reader report · {category}",
        category=category,
        run_date=run_date,
        paper_count=len(blocks),
        papers=[paper_view(block, report_path, paper_root) for block in blocks],
    )


def infer_category(blocks: list[ReaderPaperBlock]) -> str:
    if not blocks:
        return "papers"
    return blocks[0].paper.primary_category


def infer_run_date(blocks: list[ReaderPaperBlock]) -> str:
    if not blocks:
        return ""
    return blocks[0].built_date


def paper_view(block: ReaderPaperBlock, report_path: Path, paper_root: Path) -> dict[str, Any]:
    paper = block.paper
    interpretation = block.llm_interpretation
    return {
        "anchor": anchor_id(paper.arxiv_id),
        "paper": paper,
        "authors": format_authors(paper.authors),
        "abs_url": str(paper.abs_url),
        "pdf_url": str(paper.pdf_url),
        "html_url": str(paper.html_url),
        "interpretation": interpretation,
        "source": block.source,
        "figures": [key_figure_view(block, insight, report_path, paper_root) for insight in interpretation.key_figures],
    }


def key_figure_view(
    block: ReaderPaperBlock,
    insight: KeyFigureInsight,
    report_path: Path,
    paper_root: Path,
) -> dict[str, Any]:
    local_figure = find_local_figure(block, insight.index)
    article_image = block.content.images[insight.index - 1] if insight.index <= len(block.content.images) else None
    image_src = figure_image_src(block, insight.index, paper_root, local_figure)
    return {
        "index": insight.index,
        "image_src": image_src,
        "plain_caption": insight.plain_caption,
        "why_key": insight.why_key,
        "evidence": insight.evidence,
        "alt": local_figure.alt if local_figure else article_image.alt if article_image else None,
    }


def find_local_figure(block: ReaderPaperBlock, index: int) -> LocalFigure | None:
    if not block.figures:
        return None
    for figure in block.figures.figures:
        if figure.index == index:
            return figure
    return None


def figure_image_src(
    block: ReaderPaperBlock,
    index: int,
    paper_root: Path,
    local_figure: LocalFigure | None,
) -> str | None:
    if local_figure:
        paper_dir = paper_file(paper_root, block.paper.arxiv_id, "reader.json").parent
        local_path = paper_dir / local_figure.path
        if local_path.exists():
            return paper_url(block.paper.arxiv_id, local_figure.path)
        return str(local_figure.url)
    if index <= len(block.content.images):
        return str(block.content.images[index - 1].url)
    return None


def site_url(path: Path, output_root: Path) -> str:
    return "/" + path.resolve().relative_to(output_root.resolve()).as_posix()


def paper_url(arxiv_id: str, relative_path: str | Path) -> str:
    return f"/papers/{safe_arxiv_id(arxiv_id)}/{Path(relative_path).as_posix()}"


def format_authors(authors: list[str], max_authors: int = 6) -> str:
    if len(authors) <= max_authors:
        return ", ".join(authors)
    return ", ".join(authors[:max_authors]) + f", et al. ({len(authors)} authors)"


def anchor_id(arxiv_id: str) -> str:
    return "paper-" + "".join(char if char.isalnum() else "-" for char in arxiv_id).strip("-")


def render_report(context: ReportContext) -> str:
    env = Environment(
        loader=FileSystemLoader(str(template_dir())),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("report.html.j2")
    return template.render(report=context)


def template_dir() -> Path:
    return Path(str(files("arxiv_astro").joinpath("templates")))
