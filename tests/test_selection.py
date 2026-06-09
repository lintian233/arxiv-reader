from __future__ import annotations

from arxiv_astro.llm_tasks.paper_selection import PaperSelectionTask, truncate_summary
from arxiv_astro.models import PaperSelectionResult, SelectedPaper
from arxiv_astro.selection import PaperSelector, normalize_selected


class FakeSelectionLLMClient:
    model = "fake-selection-model"

    def __init__(self, selected: list[dict]) -> None:
        self.selected = selected
        self.seen_messages = []

    def chat_json(self, messages):
        self.seen_messages = messages
        return {"selected": self.selected}


def test_truncate_summary_preserves_head_and_tail() -> None:
    summary = "a" * 80 + "b" * 80 + "c" * 80

    truncated = truncate_summary(summary, max_chars=120)

    assert truncated.startswith("a")
    assert "...[truncated]..." in truncated
    assert truncated.endswith("c" * 30)


def test_paper_selection_task_builds_metadata_prompt(sample_paper) -> None:
    task = PaperSelectionTask()
    messages = task.messages(
        [sample_paper],
        interests="FRB, pulsar",
        max_results=1,
        max_input_chars=10000,
        summary_max_chars=4000,
    )

    assert messages[0]["role"] == "system"
    assert "FRB, pulsar" in messages[1]["content"]
    assert sample_paper.arxiv_id in messages[1]["content"]
    assert sample_paper.title in messages[1]["content"]
    assert "summary:" in messages[1]["content"]


def test_paper_selection_task_rejects_too_long_input(sample_paper) -> None:
    task = PaperSelectionTask()

    try:
        task.messages([sample_paper], "FRB", max_results=1, max_input_chars=10)
    except ValueError as exc:
        assert "Selection input too long" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_normalize_selected_filters_unknown_and_duplicates(sample_paper) -> None:
    selected = [
        SelectedPaper(arxiv_id="unknown", relevance=5, reason="bad"),
        SelectedPaper(arxiv_id=sample_paper.arxiv_id, relevance=4, reason="first"),
        SelectedPaper(arxiv_id=sample_paper.arxiv_id, relevance=3, reason="duplicate"),
    ]

    normalized = normalize_selected(selected, [sample_paper], max_results=2)

    assert len(normalized) == 1
    assert normalized[0].reason == "first"


def test_paper_selector_returns_selected_papers_and_block(sample_paper) -> None:
    llm = FakeSelectionLLMClient(
        [
            {
                "arxiv_id": sample_paper.arxiv_id,
                "relevance": 5,
                "matched_interests": ["FRB"],
                "reason": "matches interests",
            }
        ]
    )
    selector = PaperSelector(llm, max_input_chars=10000, summary_max_chars=4000)

    result = selector.select([sample_paper], "FRB", max_results=1, category="astro-ph", fetch_results=100)

    assert result.papers == [sample_paper]
    assert result.block.category == "astro-ph"
    assert result.block.fetch_results == 100
    assert result.block.max_results == 1
    assert result.block.selected[0].arxiv_id == sample_paper.arxiv_id
    assert result.block.llm_metadata.task == "paper_selection"
    assert result.block.llm_metadata.model == "fake-selection-model"


def test_paper_selection_result_validates_payload() -> None:
    result = PaperSelectionResult.model_validate(
        {
            "selected": [
                {
                    "arxiv_id": "2401.12345v1",
                    "relevance": 5,
                    "matched_interests": ["FRB"],
                    "reason": "relevant",
                }
            ]
        }
    )

    assert result.selected[0].relevance == 5
