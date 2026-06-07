from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import arxiv

from arxiv_astro import cli
from arxiv_astro.cli import build_parser
from arxiv_astro.models import ContentType, PaperContent
from arxiv_astro.normalize import build_paper_block, truncate_for_llm
from arxiv_astro.pipeline import Pipeline
from arxiv_astro.settings import Settings, debug_log, parse_bool, set_debug
from arxiv_astro.writer import write_jsonl, write_metadata_json, write_metadata_jsonl


class FakeArxivClient:
    def __init__(self, paper) -> None:
        self.paper = paper

    def fetch_category(self, category: str, max_results: int):
        assert category == "astro-ph.CO"
        assert max_results == 1
        return [self.paper]


class FakeContentLoader:
    def load(self, paper):
        assert paper.arxiv_id == "2401.12345v1"
        return PaperContent(content_type=ContentType.HTML, text="abcdef", text_chars=6)


class FakeLLMClient:
    def __init__(self, interpretation) -> None:
        self.interpretation = interpretation
        self.seen_text = ""

    def interpret(self, paper, text: str):
        self.seen_text = text
        return self.interpretation


def test_truncate_for_llm() -> None:
    assert truncate_for_llm("abcdef", 3) == "abc"
    assert truncate_for_llm("abc", 10) == "abc"
    assert truncate_for_llm("abc", 0) == ""


def test_pipeline_builds_blocks(sample_paper, sample_interpretation) -> None:
    llm = FakeLLMClient(sample_interpretation)
    pipeline = Pipeline(
        arxiv_client=FakeArxivClient(sample_paper),
        content_loader=FakeContentLoader(),
        llm_client=llm,
        max_input_chars=3,
    )

    blocks = pipeline.run("astro-ph.CO", 1)

    assert len(blocks) == 1
    assert llm.seen_text == "abc"
    assert blocks[0].source.used_chars == 3
    assert blocks[0].llm_interpretation.one_sentence == "一句话总结"


def test_build_paper_block_and_write_jsonl(sample_paper, sample_interpretation, tmp_path: Path) -> None:
    content = PaperContent(content_type=ContentType.ABSTRACT, text="abstract", text_chars=8)
    block = build_paper_block(sample_paper, content, sample_interpretation, "ab")

    path = write_jsonl([block], tmp_path, "astro-ph/CO", now=datetime(2024, 1, 1, 1, 2, 3))

    assert path.name == "2024-01-01_010203_astro-ph_CO.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["paper"]["arxiv_id"] == sample_paper.arxiv_id
    assert payload["source"]["used_chars"] == 2


def test_write_metadata_jsonl(sample_paper, tmp_path: Path) -> None:
    path = write_metadata_jsonl([sample_paper], tmp_path, "astro-ph/IM", now=datetime(2024, 1, 1, 1, 2, 3))

    assert path.name == "2024-01-01_010203_astro-ph_IM_metadata.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["arxiv_id"] == sample_paper.arxiv_id
    assert payload["primary_category"] == "astro-ph.CO"
    assert payload["doi"] == "10.1234/example"


def test_write_metadata_json(sample_paper, tmp_path: Path) -> None:
    path = write_metadata_json([sample_paper], tmp_path, "astro-ph/IM", now=datetime(2024, 1, 1, 1, 2, 3))

    assert path.name == "2024-01-01_010203_astro-ph_IM_metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["arxiv_id"] == sample_paper.arxiv_id
    assert payload[0]["entry_id"] == sample_paper.entry_id


def test_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "model")
    monkeypatch.setenv("OUTPUT_DIR", "out")
    monkeypatch.setenv("PDF_DIR", "pdf")
    monkeypatch.setenv("REQUEST_TIMEOUT", "12.5")
    monkeypatch.setenv("MAX_INPUT_CHARS", "123")
    monkeypatch.setenv("DEBUG", "true")

    settings = Settings.from_env()

    assert settings.api_key == "key"
    assert settings.base_url == "https://example.com"
    assert settings.request_timeout == 12.5
    assert settings.max_input_chars == 123
    assert settings.debug is True
    set_debug(False)


def test_debug_log_respects_global_switch(capsys) -> None:
    set_debug(False)
    debug_log("hidden")
    assert capsys.readouterr().err == ""

    set_debug(True)
    debug_log("visible", category="astro-ph.CO")
    assert "[DEBUG] visible category=astro-ph.CO" in capsys.readouterr().err
    set_debug(False)


def test_parse_bool() -> None:
    assert parse_bool("1") is True
    assert parse_bool("YES") is True
    assert parse_bool("off") is False
    assert parse_bool(None) is False


def test_cli_parser() -> None:
    args = build_parser().parse_args(["run", "--category", "astro-ph.CO", "--max-results", "2"])

    assert args.command == "run"
    assert args.category == "astro-ph.CO"
    assert args.max_results == 2


def test_cli_parser_supports_fetch() -> None:
    args = build_parser().parse_args(
        ["fetch", "--category", "astro-ph.IM", "--max-results", "2", "--format", "json", "--debug"]
    )

    assert args.command == "fetch"
    assert args.category == "astro-ph.IM"
    assert args.max_results == 2
    assert args.format == "json"
    assert args.debug is True


def test_cli_parser_supports_content() -> None:
    args = build_parser().parse_args(["content", "--input", "data/runs/metadata.jsonl", "--format", "json"])

    assert args.command == "content"
    assert args.input == "data/runs/metadata.jsonl"
    assert args.format == "json"


def test_cli_parser_keeps_legacy_fetch_shape() -> None:
    args = build_parser().parse_args(["--category", "astro-ph.CO", "--max-results", "2"])

    assert args.command is None
    assert args.category == "astro-ph.CO"
    assert args.max_results == 2


def test_cli_fetch_writes_metadata(monkeypatch, sample_paper, tmp_path: Path, capsys) -> None:
    class FakeFetchClient:
        def __init__(self, timeout: float) -> None:
            assert timeout == 12.5

        def fetch_category(self, category: str, max_results: int):
            assert category == "astro-ph.IM"
            assert max_results == 1
            return [sample_paper]

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("REQUEST_TIMEOUT", "12.5")
    monkeypatch.setattr(cli, "ArxivClient", FakeFetchClient)

    assert cli.main(["fetch", "--category", "astro-ph.IM", "--max-results", "1", "--debug"]) == 0

    captured = capsys.readouterr()
    output = captured.out.strip()
    assert output.endswith("_astro-ph.IM_metadata.jsonl")
    assert "[DEBUG] cli started" in captured.err
    payload = json.loads(Path(output).read_text(encoding="utf-8"))
    assert payload["entry_id"] == sample_paper.entry_id
    set_debug(False)


def test_cli_legacy_shape_defaults_to_fetch(monkeypatch, sample_paper, tmp_path: Path, capsys) -> None:
    class FakeFetchClient:
        def __init__(self, timeout: float) -> None:
            pass

        def fetch_category(self, category: str, max_results: int):
            assert category == "astro-ph.IM"
            assert max_results == 1
            return [sample_paper]

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "ArxivClient", FakeFetchClient)

    assert cli.main(["--category", "astro-ph.IM", "--max-results", "1"]) == 0

    output = capsys.readouterr().out.strip()
    assert output.endswith("_astro-ph.IM_metadata.jsonl")
    payload = json.loads(Path(output).read_text(encoding="utf-8"))
    assert payload["arxiv_id"] == sample_paper.arxiv_id


def test_cli_fetch_writes_metadata_json(monkeypatch, sample_paper, tmp_path: Path, capsys) -> None:
    class FakeFetchClient:
        def __init__(self, timeout: float) -> None:
            pass

        def fetch_category(self, category: str, max_results: int):
            return [sample_paper]

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "ArxivClient", FakeFetchClient)

    assert cli.main(["fetch", "--category", "astro-ph.IM", "--max-results", "1", "--format", "json"]) == 0

    output = capsys.readouterr().out.strip()
    assert output.endswith("_astro-ph.IM_metadata.json")
    payload = json.loads(Path(output).read_text(encoding="utf-8"))
    assert payload[0]["entry_id"] == sample_paper.entry_id


def test_cli_requires_category() -> None:
    try:
        cli.main(["fetch"])
    except SystemExit as exc:
        assert str(exc) == "--category is required"
    else:
        raise AssertionError("expected SystemExit")


def test_cli_fetch_reports_arxiv_error(monkeypatch, tmp_path: Path, capsys) -> None:
    class FailingFetchClient:
        def __init__(self, timeout: float) -> None:
            pass

        def fetch_category(self, category: str, max_results: int):
            raise arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 429)

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "ArxivClient", FailingFetchClient)

    assert cli.main(["fetch", "--category", "astro-ph.IM", "--max-results", "1"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "arXiv request failed:" in captured.err
    assert "HTTP 429" in captured.err


def test_cli_content_loads_metadata_and_writes_content(monkeypatch, sample_paper, tmp_path: Path, capsys) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(sample_paper.model_dump_json() + "\n", encoding="utf-8")

    class FakeLoader:
        def __init__(self, pdf_dir: Path, timeout: float) -> None:
            assert pdf_dir == Path("data/pdfs")

        def load(self, paper):
            return PaperContent(
                content_type=ContentType.HTML,
                text="Full text",
                text_chars=9,
                source_url=paper.html_url,
            )

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "ContentLoader", FakeLoader)

    assert cli.main(["content", "--input", str(metadata_path), "--format", "json"]) == 0

    output = capsys.readouterr().out.strip()
    assert output.endswith("_metadata_content.json")
    payload = json.loads(Path(output).read_text(encoding="utf-8"))
    assert payload[0]["content"]["text"] == "Full text"
