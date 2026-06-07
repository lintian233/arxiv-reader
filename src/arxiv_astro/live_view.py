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
    result: str = ""


class PipelineLiveRenderer:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console(stderr=True)
        self.rows: dict[str, PaperRow] = {}
        self.total = 0
        self._live: Live | None = None

    def __enter__(self) -> "PipelineLiveRenderer":
        self._live = Live(self.render(), console=self.console, refresh_per_second=4, transient=False)
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

        paper: PaperMetadata = payload["paper"]
        content: PaperContent | None = payload.get("content")
        row = self.rows.get(paper.arxiv_id) or PaperRow(
            index=payload["index"],
            total=payload["total"],
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            status=payload["status"],
        )
        row.status = payload["status"]
        if content:
            row.source = content.content_type.value
            row.text_chars = content.text_chars
            row.image_count = len(content.images)
        block: PaperBlock | None = payload.get("block")
        if block:
            row.one_sentence = block.llm_interpretation.one_sentence
            row.result = block.llm_interpretation.result
        self.rows[paper.arxiv_id] = row
        self.refresh()

    def refresh(self) -> None:
        if self._live:
            self._live.update(self.render())

    def render(self) -> Group:
        return Group(self.render_table(), self.render_interpretations())

    def render_table(self) -> Table:
        table = Table(title=f"arxiv-astro pipeline ({len(self.rows)}/{self.total})")
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("arXiv ID", no_wrap=True)
        table.add_column("Title")
        table.add_column("Status", no_wrap=True)
        table.add_column("Source", no_wrap=True)
        table.add_column("Text", justify="right", no_wrap=True)
        table.add_column("Images", justify="right", no_wrap=True)
        for row in sorted(self.rows.values(), key=lambda item: item.index):
            table.add_row(
                f"{row.index}/{row.total}",
                row.arxiv_id,
                row.title,
                STATUS_LABELS.get(row.status, row.status),
                row.source,
                str(row.text_chars or "-"),
                str(row.image_count),
            )
        return table

    def render_interpretations(self) -> Panel:
        completed_rows = [row for row in sorted(self.rows.values(), key=lambda item: item.index) if row.one_sentence or row.result]
        if not completed_rows:
            return Panel("Waiting for LLM interpretations...", title="LLM interpretations")

        body = Text()
        for row_index, row in enumerate(completed_rows):
            if row_index:
                body.append("\n\n")
            body.append(f"{row.index}/{row.total} {row.arxiv_id}  {row.title}\n", style="bold")
            if row.one_sentence:
                body.append("Summary: ", style="cyan")
                body.append(row.one_sentence)
                body.append("\n")
            if row.result:
                body.append("Result: ", style="green")
                body.append(row.result)
        return Panel(body, title="LLM interpretations")
