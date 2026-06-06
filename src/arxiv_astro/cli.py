from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arxiv
from dotenv import load_dotenv

from arxiv_astro.arxiv_client import ArxivClient
from arxiv_astro.content_loader import ContentLoader
from arxiv_astro.llm_client import LLMClient
from arxiv_astro.pipeline import Pipeline
from arxiv_astro.settings import Settings, debug_log, set_debug
from arxiv_astro.writer import write_jsonl, write_metadata_json, write_metadata_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch arXiv papers and generate LLM interpretation blocks.")
    subparsers = parser.add_subparsers(dest="command")

    fetch_parser = subparsers.add_parser("fetch", help="Fetch arXiv metadata only")
    add_common_args(fetch_parser)
    fetch_parser.add_argument("--format", choices=["jsonl", "json"], default="jsonl", help="Metadata output format")

    run_parser = subparsers.add_parser("run", help="Fetch papers and run the full LLM pipeline")
    add_common_args(run_parser)

    add_common_args(parser)
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--category", help="arXiv category, for example astro-ph.CO")
    parser.add_argument("--max-results", type=int, default=5, help="Number of latest papers to fetch")
    parser.add_argument("--debug", action="store_true", help="Print debug information to stderr")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if not args.category:
        raise SystemExit("--category is required")

    settings = Settings.from_env()
    if args.debug:
        set_debug(True)
    command = args.command or "fetch"
    debug_log("cli started", command=command, category=args.category, max_results=args.max_results)

    try:
        if command == "fetch":
            output_format = getattr(args, "format", "jsonl")
            return run_fetch(args.category, args.max_results, output_format, settings)

        return run_pipeline(args.category, args.max_results, settings)
    except arxiv.ArxivError as exc:
        print(f"arXiv request failed: {exc}", file=sys.stderr)
        return 1


def run_fetch(category: str, max_results: int, output_format: str, settings: Settings) -> int:
    papers = ArxivClient(timeout=settings.request_timeout).fetch_category(category, max_results)
    debug_log("writing metadata", format=output_format, output_dir=settings.output_dir)
    if output_format == "json":
        output_path = write_metadata_json(papers, Path(settings.output_dir), category)
    else:
        output_path = write_metadata_jsonl(papers, Path(settings.output_dir), category)
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
            timeout=settings.request_timeout,
        ),
        max_input_chars=settings.max_input_chars,
    )
    blocks = pipeline.run(category, max_results)
    output_path = write_jsonl(blocks, Path(settings.output_dir), category)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
