from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from arxiv_astro.models import PaperBlock, PaperContent, PaperMetadata


STATUS_LABELS = {
    "fetched": "[cyan]metadata[/cyan]",
    "content_loaded": "[blue]content[/blue]",
    "llm_started": "[yellow]llm[/yellow]",
    "done": "[green]done[/green]",
}


@dataclass
class PaperRow:
    index: int
    total: int
    arxiv_id: str
    title: str
    status: str
    source: str = "-"
    text_chars: int = 0
    image_count: int = 0
    one_sentence: str = ""
    main_results: str = ""
    key_figures: str = ""
    updated_at: int = 0


class PipelineLiveRenderer:
    def __init__(
        self,
        console: Console | None = None,
        max_rows: int = 8,
        max_title_chars: int = 72,
        max_preview_chars: int = 180,
    ) -> None:
        self.console = console or Console(stderr=True)
        self.max_rows = max_rows
        self.max_title_chars = max_title_chars
        self.max_preview_chars = max_preview_chars
        self.rows: dict[str, PaperRow] = {}
        self.total = 0
        self.update_count = 0
        self._live: Live | None = None

    def __enter__(self) -> "PipelineLiveRenderer":
        self._live = Live(
            self.render(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
            vertical_overflow="crop",
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._live:
            self._live.update(self.render())
            self._live.__exit__(exc_type, exc, traceback)

    def on_update(self, payload: dict) -> None:
        if payload.get("event") == "fetched":
            self.total = int(payload.get("total") or 0)
            self.refresh()
            return
        if payload.get("event") != "paper":
            return

        self.update_count += 1
        paper: PaperMetadata = payload["paper"]
        content: PaperContent | None = payload.get("content")
        row = self.rows.get(paper.arxiv_id) or PaperRow(
            index=payload["index"],
            total=payload["total"],
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            status=payload["status"],
        )
        row.updated_at = self.update_count
        row.status = payload["status"]
        if content:
            row.source = content.content_type.value
            row.text_chars = content.text_chars
            row.image_count = len(content.images)
        block: PaperBlock | None = payload.get("block")
        if block:
            row.one_sentence = block.llm_interpretation.one_sentence
            row.main_results = block.llm_interpretation.main_results
            row.key_figures = format_key_figures(block)
        self.rows[paper.arxiv_id] = row
        self.refresh()

    def refresh(self) -> None:
        if self._live:
            self._live.update(self.render())

    def render(self) -> Group:
        return Group(self.render_table(), self.render_latest_result())

    def render_table(self) -> Table:
        table = Table(title=f"arxiv-astro pipeline ({len(self.rows)}/{self.total})")
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("arXiv ID", no_wrap=True)
        table.add_column("Title")
        table.add_column("Status", no_wrap=True)
        table.add_column("Source", no_wrap=True)
        table.add_column("Text", justify="right", no_wrap=True)
        table.add_column("Images", justify="right", no_wrap=True)
        for row in self.visible_rows():
            table.add_row(
                f"{row.index}/{row.total}",
                row.arxiv_id,
                clip_text(row.title, self.max_title_chars),
                STATUS_LABELS.get(row.status, row.status),
                row.source,
                str(row.text_chars or "-"),
                str(row.image_count),
            )
        return table

    def visible_rows(self) -> list[PaperRow]:
        rows = sorted(self.rows.values(), key=lambda item: item.index)
        if self.max_rows <= 0:
            return []
        return rows[-self.max_rows :]

    def render_latest_result(self) -> Panel:
        completed_rows = [
            row
            for row in self.rows.values()
            if row.one_sentence
        ]
        if not completed_rows:
            return Panel("Waiting for LLM interpretation preview...", title="Latest result")

        row = max(completed_rows, key=lambda item: item.updated_at)
        body = Text()
        body.append(f"{row.index}/{row.total} {row.arxiv_id}  {clip_text(row.title, self.max_title_chars)}\n", style="bold")
        body.append("Summary: ", style="cyan")
        body.append(clip_text(row.one_sentence, self.max_preview_chars))
        return Panel(body, title="Latest result")


def format_key_figures(block: PaperBlock) -> str:
    return "; ".join(
        f"图{figure.index}: {figure.plain_caption}"
        for figure in block.llm_interpretation.key_figures
    )


def clip_text(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 1:
        return "…"
    return normalized[: max_chars - 1].rstrip() + "…"
