from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import sys
from pathlib import Path

import arxiv
from dotenv import load_dotenv

from arxiv_astro.arxiv_client import ArxivClient
from arxiv_astro.cli_render import (
    render_fetch_summary,
    render_next_steps,
    render_output_path,
    render_pipeline_summary,
    render_selection_summary,
    render_stage,
)
from arxiv_astro.content_io import read_content_blocks
from arxiv_astro.content_loader import ContentLoader
from arxiv_astro.figure_downloader import FigureDownloader
from arxiv_astro.llm_client import LLMClient
from arxiv_astro.live_view import PipelineLiveRenderer
from arxiv_astro.metadata_io import read_metadata
from arxiv_astro.pipeline import PaperRun, Pipeline, emit_pipeline_started
from arxiv_astro.report import generate_report
from arxiv_astro.selection import PaperSelector, SelectionError
from arxiv_astro.settings import Settings, debug_log, set_debug
from arxiv_astro.workflows import (
    build_content_context,
    build_explain_context,
    download_figure_sets,
    explain_content_blocks_with_cache,
    load_content_blocks_with_cache,
)
from arxiv_astro.writer import (
    write_selection_block,
    write_content_outputs,
    write_fetch_outputs,
    write_interpretation_outputs,
    write_reader_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read recent arXiv papers with LLM-assisted selection and interpretation.")
    subparsers = parser.add_subparsers(dest="command")

    fetch_parser = subparsers.add_parser("fetch", help="Fetch recent arXiv paper metadata")
    add_common_args(fetch_parser)

    run_parser = subparsers.add_parser("run", help="Select, read, and interpret recent arXiv papers")
    add_common_args(run_parser)
    run_parser.add_argument("--fetch-results", type=int, help="Number of recent candidates to fetch before LLM selection")
    run_parser.add_argument("--interests", help="Research interests used to select papers before full reading")

    content_parser = subparsers.add_parser("content", help="Load full text and figures from fetched metadata")
    content_parser.add_argument("--input", required=True, help="metadata.json or manifest.json path from fetch")
    content_parser.add_argument("--debug", action="store_true", help="Print debug information to stderr")

    explain_parser = subparsers.add_parser("explain", help="Generate LLM interpretations from loaded content")
    explain_parser.add_argument("--input", required=True, help="content.json or manifest.json path from content")
    explain_parser.add_argument("--debug", action="store_true", help="Print debug information to stderr")

    report_parser = subparsers.add_parser("report", help="Build a local HTML reading report")
    report_parser.add_argument("--input", required=True, help="reader.json or manifest.json path from run/explain")
    report_parser.add_argument("--debug", action="store_true", help="Print debug information to stderr")

    serve_parser = subparsers.add_parser("serve", help="Serve local reports and paper assets on localhost")
    serve_parser.add_argument("--port", type=int, default=8765, help="Local HTTP port")
    serve_parser.add_argument("--debug", action="store_true", help="Print debug information to stderr")

    add_common_args(parser)
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--category", help="arXiv category or group, e.g. astro-ph, astro-ph.IM, astro-ph.IM,astro-ph.HE")
    parser.add_argument("--max-results", type=int, default=5, help="Number of papers to read after optional selection")
    parser.add_argument("--debug", action="store_true", help="Print debug information to stderr")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    command = args.command or "fetch"
    if command not in {"content", "explain", "report", "serve"} and not args.category:
        raise SystemExit("--category is required")

    settings = Settings.from_env()
    if args.debug:
        set_debug(True)
    debug_log(
        "cli started",
        command=command,
        category=getattr(args, "category", None),
        max_results=getattr(args, "max_results", None),
    )

    try:
        if command == "fetch":
            return run_fetch(args.category, args.max_results, settings)
        if command == "content":
            return run_content(Path(args.input), settings)
        if command == "explain":
            return run_explain(Path(args.input), settings)
        if command == "report":
            return run_report(Path(args.input), settings)
        if command == "serve":
            return run_serve(settings, port=args.port)

        return run_pipeline(
            args.category,
            args.max_results,
            settings,
            fetch_results=getattr(args, "fetch_results", None),
            interests=getattr(args, "interests", None),
        )
    except arxiv.ArxivError as exc:
        print(f"arXiv request failed: {exc}", file=sys.stderr)
        return 1
    except SelectionError as exc:
        print(f"paper selection failed: {exc}", file=sys.stderr)
        return 1


def run_fetch(category: str, max_results: int, settings: Settings) -> int:
    render_stage("1/1", "Fetch recent arXiv metadata", f"category: {category}  papers: {max_results}")
    papers = ArxivClient(timeout=settings.request_timeout).fetch_category(category, max_results)
    debug_log("writing metadata", output_dir=settings.output_dir)
    output_path = write_fetch_outputs(papers, Path(settings.output_dir), category, max_results)
    render_fetch_summary(papers, category, max_results, output_path)
    render_next_steps([f"arxiv-astro content --input {output_path}"])
    print(output_path)
    return 0


def run_content(input_path: Path, settings: Settings) -> int:
    papers = read_metadata(input_path)
    debug_log("loaded metadata input", input=str(input_path), count=len(papers))
    output_root = Path(settings.output_dir)
    loader = ContentLoader(output_root=output_root, timeout=settings.request_timeout)
    blocks = load_content_blocks_with_cache(papers, loader, output_root)
    figure_sets = download_figure_sets(blocks, FigureDownloader(output_root, timeout=settings.request_timeout))
    context = build_content_context(input_path, papers)
    debug_log("writing content", output_dir=settings.output_dir)
    output_path = write_content_outputs(
        blocks,
        output_root,
        context.category,
        context.max_results,
        context.metadata_paths,
        figure_sets,
    )
    print(output_path)
    return 0


def run_explain(input_path: Path, settings: Settings) -> int:
    content_blocks = read_content_blocks(input_path)
    debug_log("loaded content input", input=str(input_path), count=len(content_blocks))
    llm_client = LLMClient(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model,
        timeout=settings.llm_request_timeout,
        max_output_tokens=settings.llm_max_output_tokens,
    )
    output_root = Path(settings.output_dir)
    blocks = explain_content_blocks_with_cache(content_blocks, llm_client, settings.max_input_chars, output_root)
    context = build_explain_context(input_path, content_blocks, output_root)
    debug_log("writing interpretations", output_dir=settings.output_dir)
    output_path = write_interpretation_outputs(
        blocks,
        context.content_by_id,
        output_root,
        context.category,
        context.max_results,
        metadata_paths=context.metadata_paths,
        content_paths=context.content_paths,
        figure_sets=context.figure_sets,
        figure_paths=context.figure_paths,
    )
    print(output_path)
    return 0


def run_pipeline(
    category: str,
    max_results: int,
    settings: Settings,
    fetch_results: int | None = None,
    interests: str | None = None,
) -> int:
    effective_interests = interests if interests is not None else settings.paper_interests
    effective_fetch_results = fetch_results if fetch_results is not None else settings.fetch_results
    debug_log(
        "running full pipeline",
        category=category,
        max_results=max_results,
        fetch_results=effective_fetch_results,
        has_interests=bool(effective_interests),
    )
    llm_client = LLMClient(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model,
        timeout=settings.llm_request_timeout,
        max_output_tokens=settings.llm_max_output_tokens,
    )
    pipeline = Pipeline(
        arxiv_client=ArxivClient(timeout=settings.request_timeout),
        content_loader=ContentLoader(output_root=Path(settings.output_dir), timeout=settings.request_timeout),
        llm_client=llm_client,
        max_input_chars=settings.max_input_chars,
        cache_root=Path(settings.output_dir),
        figure_downloader=FigureDownloader(Path(settings.output_dir), timeout=settings.request_timeout),
        paper_selector=PaperSelector(
            llm_client=llm_client,
            max_input_chars=settings.selection_max_input_chars,
            summary_max_chars=settings.selection_summary_max_chars,
        )
        if effective_interests
        else None,
    )
    candidate_count = effective_fetch_results if effective_interests and effective_fetch_results else max_results
    render_stage("1/4", "Fetch candidates", f"category: {category}  candidates: {candidate_count}")
    candidates = pipeline.fetch_candidates(category, max_results, effective_fetch_results, effective_interests)
    render_fetch_summary(
        candidates,
        category,
        candidate_count,
    )
    render_stage(
        "2/4",
        "Select papers",
        f"interests: {effective_interests or 'not set'}  requested: {max_results}",
    )
    papers = pipeline.select_papers(candidates, category, max_results, effective_fetch_results, effective_interests)
    if pipeline.selection_block:
        render_selection_summary(pipeline.selection_block, candidates)

    render_stage("3/4", "Read and interpret", f"selected papers: {len(papers)}")
    with PipelineLiveRenderer() as live:
        emit_pipeline_started(live.on_update, papers)
        blocks = [
            pipeline.process_paper(PaperRun(paper=paper, index=index, total=len(papers)), live.on_update)
            for index, paper in enumerate(papers, start=1)
        ]
    render_pipeline_summary(blocks)
    selection_path = (
        write_selection_block(pipeline.selection_block, Path(settings.output_dir))
        if pipeline.selection_block
        else None
    )
    output_path = write_reader_outputs(blocks, Path(settings.output_dir), category, max_results, selection_path=selection_path)
    render_stage("4/4", "Saved outputs", "reader manifest written")
    render_output_path("saved", output_path)
    render_next_steps(
        [
            f"arxiv-astro report --input {output_path}",
            "arxiv-astro serve --port 8765",
        ]
    )
    print(output_path)
    return 0


def run_report(input_path: Path, settings: Settings) -> int:
    debug_log("generating html report", input=str(input_path), output_dir=settings.output_dir)
    output_path = generate_report(input_path, Path(settings.output_dir))
    render_output_path("report", output_path)
    render_next_steps(["arxiv-astro serve --port 8765"])
    print(output_path)
    return 0


def run_serve(settings: Settings, port: int = 8765) -> int:
    output_root = Path(settings.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(output_root))
    server = ThreadingHTTPServer(("localhost", port), handler)
    print(f"Serving {output_root} at http://localhost:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
