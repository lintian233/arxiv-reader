from __future__ import annotations

import json
from pathlib import Path

import arxiv

from arxiv_astro import cli
from arxiv_astro.cli import build_parser
from arxiv_astro.models import ContentType, LLMInterpretation, PaperContent, PaperContentBlock
from arxiv_astro.normalize import build_paper_block, build_reader_block, truncate_for_llm
from arxiv_astro.pipeline import Pipeline
from arxiv_astro.settings import Settings, debug_log, parse_bool, set_debug
from arxiv_astro.writer import (
    write_content_block,
    write_content_outputs,
    write_fetch_outputs,
    write_interpretation_block,
    write_reader_outputs,
)


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
    assert blocks[0].content.text == "abcdef"


def test_pipeline_emits_updates(sample_paper, sample_interpretation) -> None:
    events = []
    pipeline = Pipeline(
        arxiv_client=FakeArxivClient(sample_paper),
        content_loader=FakeContentLoader(),
        llm_client=FakeLLMClient(sample_interpretation),
        max_input_chars=3,
    )

    pipeline.run("astro-ph.CO", 1, on_update=events.append)

    assert [event["event"] for event in events] == ["fetched", "paper", "paper", "paper", "paper"]
    assert [event.get("status") for event in events[1:]] == [
        "fetched",
        "content_loaded",
        "llm_started",
        "done",
    ]
    assert events[0]["total"] == 1
    assert events[-1]["content"].text_chars == 6


def test_pipeline_uses_content_and_interpretation_cache(sample_paper, sample_interpretation, tmp_path: Path) -> None:
    cached_content = PaperContent(content_type=ContentType.HTML, text="cached full text", text_chars=16)
    cached_interpretation = build_paper_block(sample_paper, cached_content, sample_interpretation, "cached")
    write_content_block(PaperContentBlock(paper=sample_paper, content=cached_content), tmp_path)
    write_interpretation_block(cached_interpretation, tmp_path)

    class FailingContentLoader:
        def load(self, paper):
            raise AssertionError("content loader should not run on cache hit")

    class FailingLLMClient:
        def interpret(self, paper, text: str):
            raise AssertionError("LLM should not run on cache hit")

    events = []
    pipeline = Pipeline(
        arxiv_client=FakeArxivClient(sample_paper),
        content_loader=FailingContentLoader(),
        llm_client=FailingLLMClient(),
        max_input_chars=20,
        cache_root=tmp_path,
    )

    blocks = pipeline.run("astro-ph.CO", 1, on_update=events.append)

    assert blocks[0].content.text == "cached full text"
    assert blocks[0].llm_interpretation.one_sentence == "一句话总结"
    assert [event.get("cache_hit") for event in events if event.get("cache_hit")] == [True, True]


def test_build_reader_block_and_write_outputs(sample_paper, sample_interpretation, tmp_path: Path) -> None:
    content = PaperContent(content_type=ContentType.ABSTRACT, text="abstract", text_chars=8)
    block = build_paper_block(sample_paper, content, sample_interpretation, "ab")
    reader = build_reader_block(content, block)

    path = write_reader_outputs([reader], tmp_path, "astro-ph.CO", max_results=1, run_date="2024-01-01")

    assert path == tmp_path / "runs" / "2024-01-01_astro-ph.CO" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    reader_path = Path(manifest["outputs"][0]["reader"])
    payload = json.loads(reader_path.read_text(encoding="utf-8"))
    assert payload["paper"]["arxiv_id"] == sample_paper.arxiv_id
    assert payload["source"]["used_chars"] == 2
    assert payload["built_date"] == "2024-01-01"


def test_write_fetch_outputs(sample_paper, tmp_path: Path) -> None:
    path = write_fetch_outputs([sample_paper], tmp_path, "astro-ph.IM", max_results=1, run_date="2024-01-01")

    assert path == tmp_path / "runs" / "2024-01-01_astro-ph.IM" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    metadata_path = Path(manifest["outputs"][0]["metadata"])
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["paper"]["primary_category"] == "astro-ph.CO"
    assert payload["paper"]["doi"] == "10.1234/example"
    assert payload["fetched_date"] == "2024-01-01"


def test_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "model")
    monkeypatch.setenv("OUTPUT_DIR", "out")
    monkeypatch.setenv("PDF_DIR", "pdf")
    monkeypatch.setenv("REQUEST_TIMEOUT", "12.5")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "180.5")
    monkeypatch.setenv("MAX_INPUT_CHARS", "123")
    monkeypatch.setenv("DEBUG", "true")

    settings = Settings.from_env()

    assert settings.api_key == "key"
    assert settings.base_url == "https://example.com"
    assert settings.output_dir == "out"
    assert settings.request_timeout == 12.5
    assert settings.llm_request_timeout == 180.5
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
    args = build_parser().parse_args(["fetch", "--category", "astro-ph.IM", "--max-results", "2", "--debug"])

    assert args.command == "fetch"
    assert args.category == "astro-ph.IM"
    assert args.max_results == 2
    assert args.debug is True


def test_cli_parser_supports_content() -> None:
    args = build_parser().parse_args(["content", "--input", "data/runs/2024-01-01_astro-ph.IM/manifest.json"])

    assert args.command == "content"
    assert args.input == "data/runs/2024-01-01_astro-ph.IM/manifest.json"


def test_cli_parser_supports_explain() -> None:
    args = build_parser().parse_args(["explain", "--input", "data/papers/2401.12345v1/content.json"])

    assert args.command == "explain"
    assert args.input == "data/papers/2401.12345v1/content.json"


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
    assert output.endswith("runs/") is False
    assert output.endswith("manifest.json")
    assert "[DEBUG] cli started" in captured.err
    manifest = json.loads(Path(output).read_text(encoding="utf-8"))
    metadata_path = Path(manifest["outputs"][0]["metadata"])
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["paper"]["entry_id"] == sample_paper.entry_id
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
    assert output.endswith("manifest.json")
    manifest = json.loads(Path(output).read_text(encoding="utf-8"))
    assert manifest["paper_ids"] == [sample_paper.arxiv_id]


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
    metadata_manifest = write_fetch_outputs([sample_paper], tmp_path, "astro-ph.IM", max_results=1, run_date="2024-01-01")

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

    assert cli.main(["content", "--input", str(metadata_manifest)]) == 0

    output = capsys.readouterr().out.strip()
    assert output.endswith("manifest.json")
    manifest = json.loads(Path(output).read_text(encoding="utf-8"))
    content_path = Path(manifest["outputs"][0]["content"])
    payload = json.loads(content_path.read_text(encoding="utf-8"))
    assert payload["content"]["text"] == "Full text"


def test_cli_content_uses_cached_content(monkeypatch, sample_paper, tmp_path: Path, capsys) -> None:
    metadata_manifest = write_fetch_outputs([sample_paper], tmp_path, "astro-ph.IM", max_results=1)
    write_content_block(
        PaperContentBlock(
            paper=sample_paper,
            content=PaperContent(content_type=ContentType.HTML, text="Cached text", text_chars=11),
        ),
        tmp_path,
    )

    class FailingLoader:
        def __init__(self, pdf_dir: Path, timeout: float) -> None:
            pass

        def load(self, paper):
            raise AssertionError("content loader should not run on cache hit")

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "ContentLoader", FailingLoader)

    assert cli.main(["content", "--input", str(metadata_manifest)]) == 0

    manifest = json.loads(Path(capsys.readouterr().out.strip()).read_text(encoding="utf-8"))
    content_path = Path(manifest["outputs"][0]["content"])
    assert json.loads(content_path.read_text(encoding="utf-8"))["content"]["text"] == "Cached text"


def test_cli_explain_loads_content_and_writes_interpretation(
    monkeypatch,
    sample_paper,
    tmp_path: Path,
    capsys,
) -> None:
    content_block = PaperContentBlock(
        paper=sample_paper,
        content=PaperContent(
            content_type=ContentType.HTML,
            text="Full text",
            text_chars=9,
            source_url=sample_paper.html_url,
        ),
    )
    content_manifest = write_content_outputs([content_block], tmp_path, "astro-ph.IM", max_results=1)

    class FakeExplainLLMClient:
        def __init__(self, api_key: str, base_url: str, model: str, timeout: float) -> None:
            assert api_key == "key"
            assert timeout == 180.0

        def interpret(self, paper, text: str):
            return LLMInterpretation(
                one_sentence="一句话",
                background="背景",
                problem="问题",
                method="方法",
                result="结果",
                importance="重要性",
                limitations="限制",
            )

    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "LLMClient", FakeExplainLLMClient)

    assert cli.main(["explain", "--input", str(content_manifest)]) == 0

    output = capsys.readouterr().out.strip()
    assert output.endswith("manifest.json")
    manifest = json.loads(Path(output).read_text(encoding="utf-8"))
    interpretation_path = Path(manifest["outputs"][0]["interpretation"])
    reader_path = Path(manifest["outputs"][0]["reader"])
    payload = json.loads(interpretation_path.read_text(encoding="utf-8"))
    assert payload["llm_interpretation"]["one_sentence"] == "一句话"
    assert payload["source"]["used_chars"] == len("Full text")
    assert json.loads(reader_path.read_text(encoding="utf-8"))["content"]["text"] == "Full text"


def test_cli_explain_uses_cached_interpretation(monkeypatch, sample_paper, sample_interpretation, tmp_path: Path, capsys) -> None:
    content_block = PaperContentBlock(
        paper=sample_paper,
        content=PaperContent(content_type=ContentType.HTML, text="Full text", text_chars=9, source_url=sample_paper.html_url),
    )
    content_manifest = write_content_outputs([content_block], tmp_path, "astro-ph.IM", max_results=1)
    write_interpretation_block(build_paper_block(sample_paper, content_block.content, sample_interpretation, "Full text"), tmp_path)

    class FailingExplainLLMClient:
        def __init__(self, api_key: str, base_url: str, model: str, timeout: float) -> None:
            pass

        def interpret(self, paper, text: str):
            raise AssertionError("LLM should not run on cache hit")

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "LLMClient", FailingExplainLLMClient)

    assert cli.main(["explain", "--input", str(content_manifest)]) == 0

    manifest = json.loads(Path(capsys.readouterr().out.strip()).read_text(encoding="utf-8"))
    interpretation_path = Path(manifest["outputs"][0]["interpretation"])
    payload = json.loads(interpretation_path.read_text(encoding="utf-8"))
    assert payload["llm_interpretation"]["one_sentence"] == "一句话总结"
