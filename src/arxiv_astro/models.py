from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl


class ContentType(StrEnum):
    HTML = "html"
    PDF = "pdf"
    ABSTRACT = "abstract"


class PaperMetadata(BaseModel):
    entry_id: str
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: datetime
    updated: datetime
    primary_category: str
    categories: list[str]
    abs_url: HttpUrl
    pdf_url: HttpUrl
    html_url: HttpUrl
    doi: str | None = None
    journal_ref: str | None = None
    comment: str | None = None


class ArticleImage(BaseModel):
    url: HttpUrl
    alt: str | None = None
    caption: str | None = None


class PaperContent(BaseModel):
    content_type: ContentType
    text: str
    text_chars: int
    source_url: HttpUrl | None = None
    images: list[ArticleImage] = Field(default_factory=list)


class SourceUsage(BaseModel):
    content_type: ContentType
    text_chars: int
    used_chars: int
    source_url: HttpUrl | None = None
    image_count: int = 0


class LLMInterpretation(BaseModel):
    one_sentence: str
    background: str
    problem: str
    method: str
    result: str
    importance: str
    limitations: str


def today_str() -> str:
    return date.today().isoformat()


class MetadataBlock(BaseModel):
    paper: PaperMetadata
    fetched_date: str = Field(default_factory=today_str)


class PaperBlock(BaseModel):
    paper: PaperMetadata
    source: SourceUsage
    llm_interpretation: LLMInterpretation
    interpreted_date: str = Field(default_factory=today_str)


class PaperContentBlock(BaseModel):
    paper: PaperMetadata
    content: PaperContent
    loaded_date: str = Field(default_factory=today_str)


class ReaderPaperBlock(BaseModel):
    paper: PaperMetadata
    content: PaperContent
    source: SourceUsage
    llm_interpretation: LLMInterpretation
    built_date: str = Field(default_factory=today_str)


class RunOutput(BaseModel):
    arxiv_id: str
    metadata: Path
    content: Path | None = None
    interpretation: Path | None = None
    reader: Path | None = None


class RunManifest(BaseModel):
    run_id: str
    category: str
    max_results: int
    run_date: str
    paper_ids: list[str]
    outputs: list[RunOutput]
