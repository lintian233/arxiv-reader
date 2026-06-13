from __future__ import annotations

from datetime import datetime, timezone

import arxiv

from arxiv_astro.arxiv_client import (
    ArxivClient,
    build_category_query,
    build_search,
    effective_page_size,
    expand_categories,
    normalize_space,
    paper_from_result,
)


class FakeArxivClient:
    def __init__(self, results: list[arxiv.Result]) -> None:
        self._results = results
        self.seen_search: arxiv.Search | None = None

    def results(self, search: arxiv.Search):
        self.seen_search = search
        return iter(self._results)


def make_result() -> arxiv.Result:
    return arxiv.Result(
        entry_id="http://arxiv.org/abs/2401.12345v1",
        updated=datetime(2024, 1, 2, tzinfo=timezone.utc),
        published=datetime(2024, 1, 1, tzinfo=timezone.utc),
        title=" A   Test\nPaper ",
        authors=[arxiv.Result.Author("Ada Lovelace"), arxiv.Result.Author("Grace Hopper")],
        summary=" A   useful\nabstract. ",
        comment=" A  comment ",
        journal_ref=" Journal  ref ",
        doi=" 10.1234/example ",
        primary_category="astro-ph.CO",
        categories=["astro-ph.CO", "astro-ph.GA"],
    )


def make_result_with_date(arxiv_id: str, published: datetime) -> arxiv.Result:
    return arxiv.Result(
        entry_id=f"http://arxiv.org/abs/{arxiv_id}",
        updated=published,
        published=published,
        title=f"Paper {arxiv_id}",
        authors=[arxiv.Result.Author("Ada Lovelace")],
        summary="A useful abstract.",
        primary_category="astro-ph.CO",
        categories=["astro-ph.CO"],
    )


def test_build_search_uses_category_and_latest_sort() -> None:
    search = build_search("astro-ph.CO", max_results=20)

    assert search.query == "cat:astro-ph.CO"
    assert search.max_results == 20
    assert search.sort_by == arxiv.SortCriterion.SubmittedDate
    assert search.sort_order == arxiv.SortOrder.Descending


def test_build_category_query_expands_astro_ph_archive() -> None:
    assert build_category_query("astro-ph") == (
        "(cat:astro-ph.CO OR cat:astro-ph.EP OR cat:astro-ph.GA OR "
        "cat:astro-ph.HE OR cat:astro-ph.IM OR cat:astro-ph.SR)"
    )


def test_build_category_query_supports_comma_separated_categories() -> None:
    assert build_category_query("astro-ph.IM, astro-ph.HE") == "(cat:astro-ph.IM OR cat:astro-ph.HE)"


def test_expand_categories_deduplicates_expanded_archives() -> None:
    assert expand_categories("astro-ph,astro-ph.IM") == [
        "astro-ph.CO",
        "astro-ph.EP",
        "astro-ph.GA",
        "astro-ph.HE",
        "astro-ph.IM",
        "astro-ph.SR",
    ]


def test_paper_from_result_normalizes_metadata() -> None:
    paper = paper_from_result(make_result())

    assert paper.entry_id == "http://arxiv.org/abs/2401.12345v1"
    assert paper.arxiv_id == "2401.12345v1"
    assert paper.title == "A Test Paper"
    assert paper.summary == "A useful abstract."
    assert paper.authors == ["Ada Lovelace", "Grace Hopper"]
    assert paper.primary_category == "astro-ph.CO"
    assert paper.categories == ["astro-ph.CO", "astro-ph.GA"]
    assert paper.doi == "10.1234/example"
    assert paper.journal_ref == "Journal ref"
    assert paper.comment == "A comment"
    assert str(paper.abs_url) == "https://arxiv.org/abs/2401.12345v1"
    assert str(paper.pdf_url) == "https://arxiv.org/pdf/2401.12345v1"
    assert str(paper.html_url) == "https://arxiv.org/html/2401.12345v1"


def test_arxiv_client_fetches_category_through_arxiv_package() -> None:
    fake_client = FakeArxivClient([make_result()])
    client = ArxivClient(client=fake_client)

    papers = client.fetch_category("astro-ph.CO", max_results=1)

    assert papers[0].arxiv_id == "2401.12345v1"
    assert fake_client.seen_search is not None
    assert fake_client.seen_search.query == "cat:astro-ph.CO"
    assert fake_client.seen_search.max_results == 1


def test_arxiv_client_fetches_latest_day_only() -> None:
    latest = datetime(2024, 1, 3, tzinfo=timezone.utc)
    previous = datetime(2024, 1, 2, tzinfo=timezone.utc)
    fake_client = FakeArxivClient(
        [
            make_result_with_date("2401.00003v1", latest),
            make_result_with_date("2401.00002v1", latest),
            make_result_with_date("2401.00001v1", previous),
        ]
    )
    client = ArxivClient(client=fake_client)

    papers = client.fetch_latest_day("astro-ph.CO", max_scan_results=200)

    assert [paper.arxiv_id for paper in papers] == ["2401.00003v1", "2401.00002v1"]
    assert fake_client.seen_search is not None
    assert fake_client.seen_search.max_results == 200


def test_effective_page_size_never_exceeds_requested_results() -> None:
    assert effective_page_size(page_size=100, max_results=2) == 2
    assert effective_page_size(page_size=10, max_results=100) == 10
    assert effective_page_size(page_size=100, max_results=0) == 1


def test_normalize_space() -> None:
    assert normalize_space(" a\n  b\tc ") == "a b c"
