from __future__ import annotations

import json

import httpx
import pytest

from arxiv_astro.llm_client import LLMClient, parse_interpretation, system_prompt, user_prompt


INTERPRETATION = {
    "one_sentence": "一句话",
    "background": "背景",
    "problem": "问题",
    "method": "方法",
    "result": "结果",
    "importance": "重要性",
    "limitations": "限制",
    "keywords": ["astro"],
    "reading_level": "入门",
    "recommended_for": ["学生"],
}


def test_llm_client_posts_openai_compatible_request(sample_paper) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer key"
        payload = json.loads(request.content)
        assert payload["model"] == "model"
        assert payload["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(INTERPRETATION)}}]},
        )

    client = LLMClient(
        api_key="key",
        base_url="https://llm.example/v1/",
        model="model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.interpret(sample_paper, "paper text")

    assert result.one_sentence == "一句话"
    assert result.keywords == ["astro"]


def test_llm_client_requires_api_key() -> None:
    with pytest.raises(ValueError):
        LLMClient(api_key="", base_url="https://example.com", model="model")


def test_llm_client_ignores_invalid_proxy_environment(monkeypatch) -> None:
    monkeypatch.setenv("NO_PROXY", "[ff00::*]")
    monkeypatch.setenv("no_proxy", "[ff00::*]")

    client = LLMClient(api_key="key", base_url="https://example.com", model="model")

    assert client.base_url == "https://example.com"


def test_prompts_and_parse_interpretation(sample_paper) -> None:
    assert "合法 JSON" in system_prompt()
    assert sample_paper.title in user_prompt(sample_paper, "正文")
    assert parse_interpretation(INTERPRETATION).reading_level == "入门"