from __future__ import annotations

from typing import Protocol

import arxiv

from arxiv_astro.models import PaperMetadata
from arxiv_astro.settings import debug_log

ARCHIVE_CATEGORY_EXPANSIONS = {
    "astro-ph": [
        "astro-ph.CO",
        "astro-ph.EP",
        "astro-ph.GA",
        "astro-ph.HE",
        "astro-ph.IM",
        "astro-ph.SR",
    ],
}


class ArxivResultClient(Protocol):
    def results(self, search: arxiv.Search): ...


class ArxivClient:
    def __init__(
        self,
        client: ArxivResultClient | None = None,
        page_size: int = 100,
        delay_seconds: float = 5.0,
        num_retries: int = 2,
        timeout: float | None = None,
    ) -> None:
        _ = timeout
        self._client = client
        self._page_size = page_size
        self._delay_seconds = delay_seconds
        self._num_retries = num_retries

    def fetch_category(self, category: str, max_results: int = 10) -> list[PaperMetadata]:
        search = build_search(category, max_results)
        client = self._client or self._build_client(max_results)
        debug_log(
            "fetching arxiv category",
            category=category,
            max_results=max_results,
            page_size=effective_page_size(self._page_size, max_results),
        )
        papers = [paper_from_result(result) for result in client.results(search)]
        debug_log("fetched arxiv papers", count=len(papers))
        return papers

    def _build_client(self, max_results: int) -> arxiv.Client:
        return arxiv.Client(
            page_size=effective_page_size(self._page_size, max_results),
            delay_seconds=self._delay_seconds,
            num_retries=self._num_retries,
        )


def paper_from_result(result: arxiv.Result) -> PaperMetadata:
    arxiv_id = result.get_short_id()
    pdf_url = result.pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
    return PaperMetadata(
        entry_id=result.entry_id,
        arxiv_id=arxiv_id,
        title=normalize_space(result.title),
        authors=[author.name for author in result.authors],
        summary=normalize_space(result.summary),
        published=result.published,
        updated=result.updated,
        primary_category=result.primary_category,
        categories=result.categories,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=pdf_url,
        html_url=f"https://arxiv.org/html/{arxiv_id}",
        doi=optional_normalize_space(result.doi),
        journal_ref=optional_normalize_space(result.journal_ref),
        comment=optional_normalize_space(result.comment),
    )


def fetch_category(
    category: str,
    max_results: int = 10,
    page_size: int = 100,
    delay_seconds: float = 3.0,
    num_retries: int = 5,
) -> list[PaperMetadata]:
    return ArxivClient(
        page_size=page_size,
        delay_seconds=delay_seconds,
        num_retries=num_retries,
    ).fetch_category(category, max_results)


def build_search(category: str, max_results: int) -> arxiv.Search:
    return arxiv.Search(
        query=build_category_query(category),
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )


def build_category_query(category: str) -> str:
    categories = expand_categories(category)
    query_parts = [f"cat:{item}" for item in categories]
    if len(query_parts) == 1:
        return query_parts[0]
    return f"({' OR '.join(query_parts)})"


def expand_categories(category: str) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for item in parse_category_input(category):
        for resolved in ARCHIVE_CATEGORY_EXPANSIONS.get(item, [item]):
            if resolved not in seen:
                expanded.append(resolved)
                seen.add(resolved)
    return expanded


def parse_category_input(category: str) -> list[str]:
    categories = [item.strip() for item in category.split(",") if item.strip()]
    if not categories:
        raise ValueError("category must not be empty")
    return categories


def effective_page_size(page_size: int, max_results: int) -> int:
    if max_results <= 0:
        return 1
    return max(1, min(page_size, max_results))


def normalize_space(value: str) -> str:
    return " ".join(value.split())


def optional_normalize_space(value: str | None) -> str | None:
    if not value:
        return None
    return normalize_space(value)
