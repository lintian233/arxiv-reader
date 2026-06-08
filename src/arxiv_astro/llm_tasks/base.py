from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel

from arxiv_astro.llm_client import LLMClient
from arxiv_astro.models import LLMMetadata


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMTaskResult(Generic[T]):
    value: T
    metadata: LLMMetadata


class LLMTask(Generic[T]):
    task_name: str
    prompt_version: str
    schema_version: str
    response_model: type[T]

    def messages(self, *args, **kwargs) -> list[dict[str, str]]:
        raise NotImplementedError

    def parse(self, raw: dict) -> T:
        return self.response_model.model_validate(raw)

    def metadata(self, llm_client: LLMClient, max_input_chars: int) -> LLMMetadata:
        return LLMMetadata(
            provider="openai-compatible",
            model=llm_client.model,
            task=self.task_name,
            prompt_version=self.prompt_version,
            schema_version=self.schema_version,
            max_input_chars=max_input_chars,
        )
