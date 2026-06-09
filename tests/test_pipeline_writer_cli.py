from __future__ import annotations

import json
from pathlib import Path

import arxiv

from arxiv_astro import cli
from arxiv_astro.cli import build_parser
from arxiv_astro.models import (
    ContentType,
    LLMInterpretation,
    LLMMetadata,
    PaperContent,
    PaperContentBlock,
    PaperSelectionSummary,
    SelectionBlock,
    SelectedPaper,
)
from arxiv_astro.normalize import build_paper_block, build_reader_block, truncate_for_llm
from arxiv_astro.pipeline import Pipeline
from arxiv_astro.selection import PaperSelector
from arxiv_astro.settings import Settings, debug_log, parse_bool, set_debug
from arxiv_astro.writer import (
    write_content_block,
    write_content_outputs,
    write_fetch_outputs,
    write_interpretation_block,
    write_reader_outputs,
    write_selection_block,
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
        self.model = "fake-model"

    def chat_json(self, messages):
        self.seen_text = messages[1]["content"]
        return self.interpretation.model_dump(mode="json")


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
    assert "用于解读的论文内容:\nabc" in llm.seen_text
    assert blocks[0].source.used_chars == 3
    assert blocks[0].llm_interpretation.one_sentence == "一句话总结"
    assert blocks[0].llm_metadata.model == "fake-model"
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
    cached_metadata = LLMMetadata(
        provider="openai-compatible",
        model="fake-model",
        task="paper_interpretation",
        prompt_version="v2",
        schema_version="v2",
        max_input_chars=20,
    )
    cached_interpretation = build_paper_block(sample_paper, cached_content, sample_interpretation, "cached", cached_metadata)
    write_content_block(PaperContentBlock(paper=sample_paper, content=cached_content), tmp_path)
    write_interpretation_block(cached_interpretation, tmp_path)

    class FailingContentLoader:
        def load(self, paper):
            raise AssertionError("content loader should not run on cache hit")

    class FailingLLMClient:
        model = "fake-model"

        def chat_json(self, messages):
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


def test_pipeline_ignores_interpretation_cache_with_mismatched_metadata(
    sample_paper,
    sample_interpretation,
    tmp_path: Path,
) -> None:
    cached_content = PaperContent(content_type=ContentType.HTML, text="cached full text", text_chars=16)
    stale_metadata = LLMMetadata(
        provider="openai-compatible",
        model="fake-model",
        task="paper_interpretation",
        prompt_version="v1",
        schema_version="v1",
        max_input_chars=20,
    )
    cached_interpretation = build_paper_block(sample_paper, cached_content, sample_interpretation, "cached", stale_metadata)
    write_content_block(PaperContentBlock(paper=sample_paper, content=cached_content), tmp_path)
    write_interpretation_block(cached_interpretation, tmp_path)

    pipeline = Pipeline(
        arxiv_client=FakeArxivClient(sample_paper),
        content_loader=FakeContentLoader(),
        llm_client=FakeLLMClient(sample_interpretation),
        max_input_chars=20,
        cache_root=tmp_path,
    )

    blocks = pipeline.run("astro-ph.CO", 1)

    assert blocks[0].llm_metadata.prompt_version == "v2"


def test_pipeline_selects_papers_before_content_and_explain(sample_paper, sample_interpretation) -> None:
    second_paper = sample_paper.model_copy(update={"arxiv_id": "2401.54321v1", "title": "Selected Paper"})

    class CandidateArxivClient:
        def __init__(self) -> None:
            self.seen_max_results = None

        def fetch_category(self, category: str, max_results: int):
            self.seen_max_results = max_results
            return [sample_paper, second_paper]

    class SeenContentLoader:
        def __init__(self) -> None:
            self.seen_ids = []

        def load(self, paper):
            self.seen_ids.append(paper.arxiv_id)
            return PaperContent(content_type=ContentType.HTML, text="selected text", text_chars=13)

    class DualLLMClient:
        model = "dual-model"

        def chat_json(self, messages):
            if "论文筛选助手" in messages[0]["content"]:
                return {
                    "selected": [
                        {
                            "arxiv_id": second_paper.arxiv_id,
                            "relevance": 5,
                            "matched_interests": ["FRB"],
                            "reason": "most relevant",
                        }
                    ]
                }
            return sample_interpretation.model_dump(mode="json")

    arxiv_client = CandidateArxivClient()
    content_loader = SeenContentLoader()
    llm_client = DualLLMClient()
    pipeline = Pipeline(
        arxiv_client=arxiv_client,
        content_loader=content_loader,
        llm_client=llm_client,
        max_input_chars=1000,
        paper_selector=PaperSelector(llm_client, max_input_chars=10000, summary_max_chars=4000),
    )

    blocks = pipeline.run("astro-ph", max_results=1, fetch_results=2, interests="FRB")

    assert arxiv_client.seen_max_results == 2
    assert content_loader.seen_ids == [second_paper.arxiv_id]
    assert blocks[0].paper.arxiv_id == second_paper.arxiv_id
    assert pipeline.selection_block.selected[0].reason == "most relevant"


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


def test_write_selection_block_and_manifest_reference(sample_paper, tmp_path: Path) -> None:
    selection = SelectionBlock(
        category="astro-ph.IM",
        fetch_results=100,
        max_results=1,
        interests="FRB",
        candidate_ids=[sample_paper.arxiv_id],
        selected=[
            SelectedPaper(
                arxiv_id=sample_paper.arxiv_id,
                relevance=5,
                matched_interests=["FRB"],
                reason="matches interests",
            )
        ],
        summary=PaperSelectionSummary(
            candidate_count=1,
            requested_count=1,
            selected_count=1,
            shortfall=0,
            shortfall_reason="",
        ),
        llm_metadata=LLMMetadata(
            provider="openai-compatible",
            model="model",
            task="paper_selection",
            prompt_version="v1",
            schema_version="v1",
            max_input_chars=220000,
        ),
    )

    selection_path = write_selection_block(selection, tmp_path, run_date="2024-01-01")
    manifest_path = write_fetch_outputs(
        [sample_paper],
        tmp_path,
        "astro-ph.IM",
        max_results=1,
        run_date="2024-01-01",
        selection_path=selection_path,
    )

    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert selection_path == tmp_path / "runs" / "2024-01-01_astro-ph.IM" / "selection.json"
    assert selection_payload["selected"][0]["reason"] == "matches interests"
    assert selection_payload["summary"]["candidate_count"] == 1
    assert selection_payload["summary"]["shortfall"] == 0
    assert selection_payload["selection_date"] == "2024-01-01"
    assert manifest_payload["selection"] == str(selection_path)


def test_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "model")
    monkeypatch.setenv("OUTPUT_DIR", "out")
    monkeypatch.setenv("REQUEST_TIMEOUT", "12.5")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "180.5")
    monkeypatch.setenv("MAX_INPUT_CHARS", "123")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "456")
    monkeypatch.setenv("PAPER_INTERESTS", "FRB")
    monkeypatch.setenv("FETCH_RESULTS", "99")
    monkeypatch.setenv("SELECTION_MAX_INPUT_CHARS", "789")
    monkeypatch.setenv("SELECTION_SUMMARY_MAX_CHARS", "321")
    monkeypatch.setenv("DEBUG", "true")

    settings = Settings.from_env()

    assert settings.api_key == "key"
    assert settings.base_url == "https://example.com"
    assert settings.output_dir == "out"
    assert settings.request_timeout == 12.5
    assert settings.llm_request_timeout == 180.5
    assert settings.max_input_chars == 123
    assert settings.llm_max_output_tokens == 456
    assert settings.paper_interests == "FRB"
    assert settings.fetch_results == 99
    assert settings.selection_max_input_chars == 789
    assert settings.selection_summary_max_chars == 321
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
    args = build_parser().parse_args(
        [
            "run",
            "--category",
            "astro-ph.CO",
            "--fetch-results",
            "100",
            "--max-results",
            "2",
            "--interests",
            "FRB",
        ]
    )

    assert args.command == "run"
    assert args.category == "astro-ph.CO"
    assert args.fetch_results == 100
    assert args.max_results == 2
    assert args.interests == "FRB"


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


def test_cli_parser_supports_report() -> None:
    args = build_parser().parse_args(["report", "--input", "data/runs/2024-01-01_astro-ph.IM/manifest.json"])

    assert args.command == "report"
    assert args.input == "data/runs/2024-01-01_astro-ph.IM/manifest.json"


def test_cli_parser_supports_serve() -> None:
    args = build_parser().parse_args(["serve", "--port", "8766"])

    assert args.command == "serve"
    assert args.port == 8766


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
        def __init__(self, output_root: Path, timeout: float) -> None:
            assert output_root == tmp_path

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
        def __init__(self, output_root: Path, timeout: float) -> None:
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
        def __init__(
            self,
            api_key: str,
            base_url: str,
            model: str,
            timeout: float,
            max_output_tokens: int,
        ) -> None:
            assert api_key == "key"
            assert timeout == 180.0
            assert max_output_tokens == 12000
            self.model = model

        def chat_json(self, messages):
            return {
                "one_sentence": "一句话",
                "problem_context": "问题背景",
                "why_it_matters": "为什么重要",
                "what_the_paper_does": "做了什么",
                "main_results": "核心结果",
                "key_figures": [],
                "limitations": "限制",
                "field_position": "领域位置",
            }

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
    assert payload["llm_metadata"]["task"] == "paper_interpretation"
    assert payload["source"]["used_chars"] == len("Full text")
    assert json.loads(reader_path.read_text(encoding="utf-8"))["content"]["text"] == "Full text"


def test_cli_explain_uses_cached_interpretation(monkeypatch, sample_paper, sample_interpretation, tmp_path: Path, capsys) -> None:
    content_block = PaperContentBlock(
        paper=sample_paper,
        content=PaperContent(content_type=ContentType.HTML, text="Full text", text_chars=9, source_url=sample_paper.html_url),
    )
    content_manifest = write_content_outputs([content_block], tmp_path, "astro-ph.IM", max_results=1)
    llm_metadata = LLMMetadata(
        provider="openai-compatible",
        model="deepseek-v4-pro",
        task="paper_interpretation",
        prompt_version="v2",
        schema_version="v2",
        max_input_chars=400000,
    )
    write_interpretation_block(
        build_paper_block(sample_paper, content_block.content, sample_interpretation, "Full text", llm_metadata),
        tmp_path,
    )

    class FailingExplainLLMClient:
        def __init__(
            self,
            api_key: str,
            base_url: str,
            model: str,
            timeout: float,
            max_output_tokens: int,
        ) -> None:
            assert max_output_tokens == 12000
            self.model = model

        def chat_json(self, messages):
            raise AssertionError("LLM should not run on cache hit")

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "LLMClient", FailingExplainLLMClient)

    assert cli.main(["explain", "--input", str(content_manifest)]) == 0

    manifest = json.loads(Path(capsys.readouterr().out.strip()).read_text(encoding="utf-8"))
    interpretation_path = Path(manifest["outputs"][0]["interpretation"])
    payload = json.loads(interpretation_path.read_text(encoding="utf-8"))
    assert payload["llm_interpretation"]["one_sentence"] == "一句话总结"


def test_cli_report_generates_html(monkeypatch, sample_paper, sample_interpretation, tmp_path: Path, capsys) -> None:
    content = PaperContent(content_type=ContentType.ABSTRACT, text="abstract", text_chars=8)
    reader = build_reader_block(content, build_paper_block(sample_paper, content, sample_interpretation, "abstract"))
    manifest_path = write_reader_outputs([reader], tmp_path, "astro-ph.IM", max_results=1, run_date="2024-01-01")

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    assert cli.main(["report", "--input", str(manifest_path)]) == 0

    output = Path(capsys.readouterr().out.strip())
    assert output == tmp_path / "runs" / "2024-01-01_astro-ph.IM" / "report.html"
    assert sample_paper.title in output.read_text(encoding="utf-8")


def test_cli_run_reports_selection_error(monkeypatch, tmp_path: Path, capsys) -> None:
    class FailingPipeline:
        selection_block = None

        def __init__(self, **kwargs) -> None:
            pass

        def fetch_candidates(self, *args, **kwargs):
            from arxiv_astro.selection import SelectionError

            raise SelectionError("selection failed")

    class FakeLLMClient:
        def __init__(self, **kwargs) -> None:
            self.model = kwargs["model"]

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setattr(cli, "Pipeline", FailingPipeline)
    monkeypatch.setattr(cli, "LLMClient", FakeLLMClient)

    assert cli.main(["run", "--category", "astro-ph", "--interests", "FRB"]) == 1

    assert "paper selection failed: selection failed" in capsys.readouterr().err


def test_cli_serve_uses_output_dir_and_port(monkeypatch, tmp_path: Path, capsys) -> None:
    seen = {}

    class FakeServer:
        def __init__(self, address, handler) -> None:
            seen["address"] = address
            seen["handler"] = handler
            seen["closed"] = False

        def serve_forever(self) -> None:
            seen["served"] = True

        def server_close(self) -> None:
            seen["closed"] = True

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "ThreadingHTTPServer", FakeServer)

    assert cli.main(["serve", "--port", "8766"]) == 0

    assert seen["address"] == ("localhost", 8766)
    assert seen["served"] is True
    assert seen["closed"] is True
    assert f"Serving {tmp_path.resolve()} at http://localhost:8766/" in capsys.readouterr().out
