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


def today_str() -> str:
    return date.today().isoformat()


class LocalFigure(BaseModel):
    index: int
    url: HttpUrl
    path: Path
    alt: str | None = None
    caption: str | None = None


class FigureSet(BaseModel):
    arxiv_id: str
    figures: list[LocalFigure] = Field(default_factory=list)
    downloaded_date: str = Field(default_factory=today_str)


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


class KeyFigureInsight(BaseModel):
    index: int = Field(ge=1)
    plain_caption: str
    why_key: str
    evidence: str | None = None


class LLMInterpretation(BaseModel):
    one_sentence: str
    problem_context: str
    why_it_matters: str
    what_the_paper_does: str
    main_results: str
    key_figures: list[KeyFigureInsight] = Field(default_factory=list)
    limitations: str
    field_position: str


class LLMMetadata(BaseModel):
    provider: str
    model: str
    task: str
    prompt_version: str
    schema_version: str
    max_input_chars: int


class SelectedPaper(BaseModel):
    arxiv_id: str
    relevance: int = Field(ge=1, le=5)
    matched_interests: list[str] = Field(default_factory=list)
    reason: str


class PaperSelectionResult(BaseModel):
    selected: list[SelectedPaper] = Field(default_factory=list)
    shortfall_reason: str = ""


class PaperSelectionSummary(BaseModel):
    candidate_count: int
    requested_count: int
    selected_count: int
    shortfall: int
    shortfall_reason: str = ""


class SelectionBlock(BaseModel):
    category: str
    fetch_results: int
    max_results: int
    interests: str
    candidate_ids: list[str]
    selected: list[SelectedPaper]
    summary: PaperSelectionSummary
    llm_metadata: LLMMetadata
    selection_date: str = Field(default_factory=today_str)


class MetadataBlock(BaseModel):
    paper: PaperMetadata
    fetched_date: str = Field(default_factory=today_str)


class PaperBlock(BaseModel):
    paper: PaperMetadata
    source: SourceUsage
    llm_interpretation: LLMInterpretation
    llm_metadata: LLMMetadata | None = None
    interpreted_date: str = Field(default_factory=today_str)


class PaperContentBlock(BaseModel):
    paper: PaperMetadata
    content: PaperContent
    loaded_date: str = Field(default_factory=today_str)


class ReaderPaperBlock(BaseModel):
    paper: PaperMetadata
    content: PaperContent
    figures: FigureSet | None = None
    source: SourceUsage
    llm_interpretation: LLMInterpretation
    llm_metadata: LLMMetadata | None = None
    built_date: str = Field(default_factory=today_str)


class RunOutput(BaseModel):
    arxiv_id: str
    metadata: Path
    content: Path | None = None
    figures: Path | None = None
    interpretation: Path | None = None
    reader: Path | None = None


class RunManifest(BaseModel):
    run_id: str
    category: str
    max_results: int
    run_date: str
    paper_ids: list[str]
    outputs: list[RunOutput]
    selection: Path | None = None


class PaperStatus(BaseModel):
    arxiv_id: str
    title: str | None = None
    primary_category: str | None = None
    has_metadata: bool = False
    has_content: bool = False
    has_figures: bool = False
    has_interpretation: bool = False
    has_reader: bool = False
    has_pdf: bool = False
