from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import posixpath
from urllib.parse import unquote, urlsplit


class ReaderRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, paper_root: Path, runs_root: Path, **kwargs) -> None:
        self.paper_root = paper_root.resolve()
        self.runs_root = runs_root.resolve()
        super().__init__(*args, directory=str(self.runs_root), **kwargs)

    def translate_path(self, path: str) -> str:
        route, relative_path = split_route(path)
        if route == "papers":
            return str(safe_join(self.paper_root / "papers", relative_path))
        if route == "runs":
            return str(safe_join(self.runs_root / "runs", relative_path))
        return str(safe_join(self.runs_root, relative_path))


def split_route(path: str) -> tuple[str | None, str]:
    parsed_path = unquote(urlsplit(path).path)
    normalized = posixpath.normpath(parsed_path)
    parts = [part for part in normalized.split("/") if part]
    if parts and parts[0] in {"papers", "runs"}:
        return parts[0], "/".join(parts[1:])
    return None, "/".join(parts)


def safe_join(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate == root or root in candidate.parents:
        return candidate
    return root / "__not_found__"


def serve_local(paper_root: Path, runs_root: Path, port: int = 8765) -> None:
    paper_root.resolve().mkdir(parents=True, exist_ok=True)
    runs_root.resolve().mkdir(parents=True, exist_ok=True)
    handler = partial(ReaderRequestHandler, paper_root=paper_root, runs_root=runs_root)
    server = ThreadingHTTPServer(("localhost", port), handler)
    print(f"Serving runs {runs_root.resolve()} and papers {paper_root.resolve() / 'papers'} at http://localhost:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        server.server_close()
