from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from arxiv_astro.llm_client import LLMClient, extra_body
from arxiv_astro.llm_tasks.paper_interpretation import (
    PaperInterpretationTask,
    parse_interpretation,
    system_prompt,
    user_prompt,
)


INTERPRETATION = {
    "one_sentence": "一句话",
    "problem_context": "问题背景",
    "why_it_matters": "为什么重要",
    "what_the_paper_does": "做了什么",
    "main_results": "核心结果",
    "key_figures": [{"index": 1, "plain_caption": "图展示趋势", "why_key": "支撑核心结果", "evidence": None}],
    "limitations": "限制",
    "field_position": "领域位置",
}


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(INTERPRETATION)),
                )
            ]
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_llm_client_uses_openai_sdk_request(sample_paper) -> None:
    fake_client = FakeOpenAIClient()

    client = LLMClient(
        api_key="key",
        base_url="https://llm.example/v1/",
        model="model",
        client=fake_client,
    )

    result = client.chat_json(PaperInterpretationTask().messages(sample_paper, "paper text"))

    request = fake_client.completions.kwargs
    assert request["model"] == "model"
    assert request["stream"] is False
    assert request["reasoning_effort"] == "high"
    # assert request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][1]["role"] == "user"
    assert "paper text" in request["messages"][1]["content"]
    assert request["response_format"] == {"type": "json_object"}
    assert request["max_tokens"] == 12000
    assert result["one_sentence"] == "一句话"


def test_llm_client_requires_api_key() -> None:
    with pytest.raises(ValueError):
        LLMClient(api_key="", base_url="https://example.com", model="model")


def test_llm_client_ignores_invalid_proxy_environment(monkeypatch) -> None:
    monkeypatch.setenv("NO_PROXY", "[ff00::*]")
    monkeypatch.setenv("no_proxy", "[ff00::*]")

    client = LLMClient(api_key="key", base_url="https://example.com", model="model")

    assert client.base_url == "https://example.com"


def test_extra_body_can_disable_thinking() -> None:
    assert extra_body(True) == {"thinking": {"type": "enabled"}}
    assert extra_body(False) == {}


def test_prompts_and_parse_interpretation(sample_paper) -> None:
    assert "合法 JSON" in system_prompt()
    assert "one_sentence" in system_prompt()
    assert "key_figures" in system_prompt()
    assert "不要添加额外字段" in system_prompt()
    assert sample_paper.title in user_prompt(sample_paper, "正文")
    assert parse_interpretation(INTERPRETATION).one_sentence == "一句话"


def test_paper_interpretation_task_returns_metadata(sample_paper) -> None:
    class FakeLLMClient:
        model = "model"

        def chat_json(self, messages):
            assert "正文" in messages[1]["content"]
            return INTERPRETATION

    result = PaperInterpretationTask().run(FakeLLMClient(), sample_paper, "正文", max_input_chars=123)

    assert result.value.one_sentence == "一句话"
    assert result.metadata.task == "paper_interpretation"
    assert result.value.key_figures[0].index == 1
    assert result.metadata.prompt_version == "v2"
    assert result.metadata.schema_version == "v2"
    assert result.metadata.max_input_chars == 123
