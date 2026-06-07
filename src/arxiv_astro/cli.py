from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arxiv
from dotenv import load_dotenv

from arxiv_astro.arxiv_client import ArxivClient
from arxiv_astro.content_io import content_paths_from_manifest, read_content_blocks
from arxiv_astro.content_pipeline import load_content_blocks
from arxiv_astro.content_loader import ContentLoader
from arxiv_astro.explain_pipeline import explain_content_blocks
from arxiv_astro.llm_client import LLMClient
from arxiv_astro.live_view import PipelineLiveRenderer
from arxiv_astro.metadata_io import manifest_context, metadata_paths_from_manifest, read_metadata
from arxiv_astro.pipeline import Pipeline
from arxiv_astro.settings import Settings, debug_log, set_debug
from arxiv_astro.writer import (
    write_content_outputs,
    write_fetch_outputs,
    write_interpretation_outputs,
    write_reader_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch arXiv papers and generate LLM interpretation blocks.")
    subparsers = parser.add_subparsers(dest="command")

    fetch_parser = subparsers.add_parser("fetch", help="Fetch arXiv metadata only")
    add_common_args(fetch_parser)

    run_parser = subparsers.add_parser("run", help="Fetch papers and run the full LLM pipeline")
    add_common_args(run_parser)

    content_parser = subparsers.add_parser("content", help="Load full text and images from metadata")
    content_parser.add_argument("--input", required=True, help="metadata.json or manifest.json path from fetch")
    content_parser.add_argument("--debug", action="store_true", help="Print debug information to stderr")

    explain_parser = subparsers.add_parser("explain", help="Generate LLM interpretations from content")
    explain_parser.add_argument("--input", required=True, help="content.json or manifest.json path from content")
    explain_parser.add_argument("--debug", action="store_true", help="Print debug information to stderr")

    add_common_args(parser)
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--category", help="arXiv category, for example astro-ph.CO")
    parser.add_argument("--max-results", type=int, default=5, help="Number of latest papers to fetch")
    parser.add_argument("--debug", action="store_true", help="Print debug information to stderr")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    command = args.command or "fetch"
    if command not in {"content", "explain"} and not args.category:
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

        return run_pipeline(args.category, args.max_results, settings)
    except arxiv.ArxivError as exc:
        print(f"arXiv request failed: {exc}", file=sys.stderr)
        return 1


def run_fetch(category: str, max_results: int, settings: Settings) -> int:
    papers = ArxivClient(timeout=settings.request_timeout).fetch_category(category, max_results)
    debug_log("writing metadata", output_dir=settings.output_dir)
    output_path = write_fetch_outputs(papers, Path(settings.output_dir), category, max_results)
    print(output_path)
    return 0


def run_content(input_path: Path, settings: Settings) -> int:
    papers = read_metadata(input_path)
    debug_log("loaded metadata input", input=str(input_path), count=len(papers))
    loader = ContentLoader(pdf_dir=Path(settings.pdf_dir), timeout=settings.request_timeout)
    blocks = load_content_blocks(papers, loader)
    if input_path.name == "manifest.json":
        category, max_results = manifest_context(input_path)
        metadata_paths = metadata_paths_from_manifest(input_path)
    else:
        category, max_results = papers[0].primary_category, len(papers)
        metadata_paths = None
    debug_log("writing content", output_dir=settings.output_dir)
    output_path = write_content_outputs(blocks, Path(settings.output_dir), category, max_results, metadata_paths)
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
    )
    blocks = explain_content_blocks(content_blocks, llm_client, settings.max_input_chars)
    if input_path.name == "manifest.json":
        category, max_results = manifest_context(input_path)
        content_paths = content_paths_from_manifest(input_path)
        metadata_paths = metadata_paths_from_manifest(input_path)
    else:
        category, max_results = content_blocks[0].paper.primary_category, len(content_blocks)
        content_paths = None
        metadata_paths = None
    content_by_id = {block.paper.arxiv_id: block for block in content_blocks}
    debug_log("writing interpretations", output_dir=settings.output_dir)
    output_path = write_interpretation_outputs(
        blocks,
        content_by_id,
        Path(settings.output_dir),
        category,
        max_results,
        metadata_paths=metadata_paths,
        content_paths=content_paths,
    )
    print(output_path)
    return 0


def run_pipeline(category: str, max_results: int, settings: Settings) -> int:
    debug_log("running full pipeline", category=category, max_results=max_results)
    pipeline = Pipeline(
        arxiv_client=ArxivClient(timeout=settings.request_timeout),
        content_loader=ContentLoader(pdf_dir=Path(settings.pdf_dir), timeout=settings.request_timeout),
        llm_client=LLMClient(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.model,
            timeout=settings.llm_request_timeout,
        ),
        max_input_chars=settings.max_input_chars,
    )
    with PipelineLiveRenderer() as live:
        blocks = pipeline.run(category, max_results, on_update=live.on_update)
    output_path = write_reader_outputs(blocks, Path(settings.output_dir), category, max_results)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
