from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

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


class LLMInterpretation(BaseModel):
    one_sentence: str
    background: str
    problem: str
    method: str
    result: str
    importance: str
    limitations: str
    keywords: list[str] = Field(default_factory=list)
    reading_level: str
    recommended_for: list[str] = Field(default_factory=list)


class PaperBlock(BaseModel):
    paper: PaperMetadata
    source: SourceUsage
    llm_interpretation: LLMInterpretation
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PaperContentBlock(BaseModel):
    paper: PaperMetadata
    content: PaperContent
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
