from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from arxiv_astro.models import PaperMetadata, ReaderPaperBlock, SelectionBlock


def default_console() -> Console:
    return Console(stderr=True)


def render_fetch_summary(
    papers: list[PaperMetadata],
    category: str,
    max_results: int,
    manifest_path: Path | None = None,
    console: Console | None = None,
) -> None:
    target = console or default_console()
    table = Table(title=f"arxiv-astro fetch: {category} ({len(papers)}/{max_results})")
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("arXiv ID", no_wrap=True)
    table.add_column("Category", no_wrap=True)
    table.add_column("Published", no_wrap=True)
    table.add_column("Title")
    for index, paper in enumerate(papers, start=1):
        table.add_row(
            str(index),
            paper.arxiv_id,
            paper.primary_category,
            paper.published.date().isoformat(),
            paper.title,
        )
    target.print(table)
    if manifest_path:
        target.print(Text(f"saved: {manifest_path}", style="dim"))


def render_selection_summary(
    selection: SelectionBlock,
    papers: list[ReaderPaperBlock] | list[PaperMetadata],
    console: Console | None = None,
) -> None:
    target = console or default_console()
    titles = paper_titles_by_id(papers)
    table = Table(title=f"LLM selection: {selection.summary.selected_count}/{selection.summary.requested_count}")
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("arXiv ID", no_wrap=True)
    table.add_column("Relevance", justify="right", no_wrap=True)
    table.add_column("Matched")
    table.add_column("Title")
    for index, selected in enumerate(selection.selected, start=1):
        table.add_row(
            str(index),
            selected.arxiv_id,
            str(selected.relevance),
            ", ".join(selected.matched_interests) or "-",
            titles.get(selected.arxiv_id, "-"),
        )
    target.print(table)
    target.print(
        Text(
            "candidates: "
            f"{selection.summary.candidate_count}  "
            f"requested: {selection.summary.requested_count}  "
            f"selected: {selection.summary.selected_count}  "
            f"shortfall: {selection.summary.shortfall}",
            style="dim",
        )
    )
    if selection.summary.shortfall_reason:
        target.print(Text(f"shortfall reason: {selection.summary.shortfall_reason}", style="dim"))
    for selected in selection.selected:
        if selected.reason:
            target.print(Text(f"{selected.arxiv_id}: {selected.reason}", style="dim"))


def render_output_path(label: str, path: Path, console: Console | None = None) -> None:
    target = console or default_console()
    target.print(Text(f"{label}: {path}", style="dim"))


def render_pipeline_summary(
    blocks: list[ReaderPaperBlock],
    console: Console | None = None,
) -> None:
    target = console or default_console()
    render_pipeline_final_notes(blocks, target)


def render_pipeline_final_notes(blocks: list[ReaderPaperBlock], console: Console) -> None:
    if not blocks:
        return
    console.print()
    for index, block in enumerate(blocks, start=1):
        if index > 1:
            console.print()
        console.print(
            Text(f"{index}/{len(blocks)} {block.paper.arxiv_id}  {block.paper.title}", style="bold")
        )
        console.print(Text("Summary:", style="cyan"))
        console.print(block.llm_interpretation.one_sentence)
        console.print(Text("Core result:", style="green"))
        console.print(block.llm_interpretation.main_results)
        key_figures = format_key_figures(block)
        if key_figures:
            console.print(Text("Key figures:", style="magenta"))
            console.print(key_figures)


def paper_titles_by_id(papers: list[ReaderPaperBlock] | list[PaperMetadata]) -> dict[str, str]:
    titles: dict[str, str] = {}
    for item in papers:
        paper = item if isinstance(item, PaperMetadata) else item.paper
        titles[paper.arxiv_id] = paper.title
    return titles


def format_key_figures(block: ReaderPaperBlock) -> str:
    return "；".join(
        f"图{figure.index}: {figure.plain_caption}"
        for figure in block.llm_interpretation.key_figures
    )
