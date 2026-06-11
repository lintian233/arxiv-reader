from __future__ import annotations

from pathlib import Path

from arxiv_astro.local_server import safe_join, split_route


def test_split_route_maps_papers_and_runs() -> None:
    assert split_route("/papers/2401.1/figures/a.png") == ("papers", "2401.1/figures/a.png")
    assert split_route("/runs/2024/report.html") == ("runs", "2024/report.html")
    assert split_route("/index.html") == (None, "index.html")


def test_split_route_normalizes_encoded_paths() -> None:
    assert split_route("/papers/2401.1/../2401.2/a%20b.png") == ("papers", "2401.2/a b.png")


def test_safe_join_rejects_path_traversal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    assert safe_join(root, "child/file.txt") == root / "child" / "file.txt"
    assert safe_join(root, "../outside.txt") == root / "__not_found__"
