import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator

RetrievalMode = Literal["semantic", "keyword"]


class IndexedDocument(BaseModel):
    id: str
    name: str
    chunk_count: int = Field(ge=1)
    source_type: Literal["local", "url", "unknown"] | None = None


class IndexResponse(BaseModel):
    collection: str
    status: Literal["ready", "empty", "partial"]
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    documents: list[IndexedDocument]
    truncated: bool = False


class IndexErrorResponse(BaseModel):
    status: Literal["unavailable"] = "unavailable"
    message: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)
    retrieval_mode: RetrievalMode = "semantic"

    @field_validator("question", mode="before")
    @classmethod
    def trim_question(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class QuerySearchResult(BaseModel):
    rank: int = Field(ge=1)
    id: str
    source: str
    score: float
    content: str
    matched_terms: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("matched_terms")
    @classmethod
    def matched_terms_are_bounded(cls, value: list[str]) -> list[str]:
        if any(not term or len(term) > 64 for term in value):
            raise ValueError("Matched terms must be short non-empty text")
        return value


class QuerySearchResponse(BaseModel):
    mode: RetrievalMode
    label: str
    technical_label: str
    score_label: str
    results: list[QuerySearchResult]


class QueryPromptMessage(BaseModel):
    role: Literal["system", "human"]
    content: str


class QueryPromptResponse(BaseModel):
    label: str
    technical_label: str
    note: str
    messages: list[QueryPromptMessage]


class QueryVisualizationPoint(BaseModel):
    id: str
    source: str
    x: float
    y: float
    z: float
    retrieved: bool
    rank: int | None = Field(default=None, ge=1)
    preview: str | None = Field(default=None, max_length=240)

    @field_validator("x", "y", "z")
    @classmethod
    def coordinates_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Visualization coordinates must be finite")
        return value


class QueryVisualizationQueryPoint(BaseModel):
    x: float
    y: float
    z: float

    @field_validator("x", "y", "z")
    @classmethod
    def coordinates_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Visualization coordinates must be finite")
        return value


class QueryVisualizationResponse(BaseModel):
    status: Literal["ready", "partial", "unavailable", "unsupported"]
    label: str
    technical_label: str
    note: str
    points: list[QueryVisualizationPoint]
    query: QueryVisualizationQueryPoint | None
    truncated: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    search: QuerySearchResponse
    visualization: QueryVisualizationResponse
    prompt: QueryPromptResponse


class QueryErrorResponse(BaseModel):
    status: Literal["unavailable"] = "unavailable"
    message: str
