import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RetrievalMode = Literal["semantic", "keyword", "hybrid"]
ProjectionMethod = Literal["pca", "tsne", "umap"]

MAX_PUBLIC_CHUNK_CHARS = 8_000


MAX_EXAMPLE_QUESTION_SETS = 20
MAX_EXAMPLE_QUESTIONS = 50
MAX_EXAMPLE_QUESTION_CHARS = 500
MAX_EXAMPLE_SET_ID_CHARS = 40
_EXAMPLE_SET_ID = re.compile(r"[a-z][a-z0-9_]*")


class ExampleQuestionSet(BaseModel):
    id: str = Field(max_length=MAX_EXAMPLE_SET_ID_CHARS)
    questions: list[str] = Field(min_length=1, max_length=MAX_EXAMPLE_QUESTIONS)

    @field_validator("id", mode="before")
    @classmethod
    def validate_id(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("Question set IDs must be text")
        normalized = value.strip().casefold()
        if not _EXAMPLE_SET_ID.fullmatch(normalized):
            raise ValueError("Question set IDs must use lowercase letters, numbers, and underscores")
        return normalized

    @field_validator("questions", mode="before")
    @classmethod
    def validate_questions(cls, value: object) -> list[str]:
        if not isinstance(value, list) or not 1 <= len(value) <= MAX_EXAMPLE_QUESTIONS:
            raise ValueError("Expected 1–50 questions in each set")
        questions: list[str] = []
        for question in value:
            if not isinstance(question, str):
                raise ValueError("Questions must be text")
            question = question.strip()
            if not 1 <= len(question) <= MAX_EXAMPLE_QUESTION_CHARS:
                raise ValueError("Questions must contain 1–500 characters")
            questions.append(question)
        return questions


class ExampleQuestions(BaseModel):
    display_count: int = Field(strict=True, ge=1)
    question_sets: list[ExampleQuestionSet] = Field(
        min_length=1,
        max_length=MAX_EXAMPLE_QUESTION_SETS,
    )

    @model_validator(mode="after")
    def validate_config(self) -> "ExampleQuestions":
        set_ids = [question_set.id for question_set in self.question_sets]
        if len(set(set_ids)) != len(set_ids):
            raise ValueError("Question set IDs must be unique")

        questions = [
            question
            for question_set in self.question_sets
            for question in question_set.questions
        ]
        if len(questions) > MAX_EXAMPLE_QUESTIONS:
            raise ValueError("Expected no more than 50 questions in total")
        normalized_questions = [" ".join(question.split()).casefold() for question in questions]
        if len(set(normalized_questions)) != len(normalized_questions):
            raise ValueError("Questions must be unique across all sets")
        if self.display_count > len(questions):
            raise ValueError("Display count exceeds total question count")
        return self


class PublicExampleQuestionSet(BaseModel):
    questions: list[str]


class PublicExampleQuestions(BaseModel):
    display_count: int
    question_sets: list[PublicExampleQuestionSet]


class IndexedChunk(BaseModel):
    id: str
    content: str = Field(max_length=MAX_PUBLIC_CHUNK_CHARS)
    # Display numbering only: ingestion does not persist original source order.
    display_order: int = Field(ge=1)
    truncated: bool = False


class IndexedDocumentChunksResponse(BaseModel):
    document_id: str
    document_name: str
    chunks: list[IndexedChunk]
    truncated: bool = False


class PublicChunkDetails(BaseModel):
    id: str
    source: str
    content: str = Field(max_length=MAX_PUBLIC_CHUNK_CHARS)
    truncated: bool = False


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


class RetrievalCapabilitiesResponse(BaseModel):
    retrieval_modes: list[RetrievalMode]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)
    retrieval_mode: RetrievalMode = "semantic"

    @field_validator("question", mode="before")
    @classmethod
    def trim_question(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class AggregateQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1_000)

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
    similarity: float
    preview: str | None = Field(default=None, max_length=240)

    @field_validator("x", "y", "z", "similarity")
    @classmethod
    def values_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Visualization values must be finite")
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


class QueryVisualizationCoordinate(QueryVisualizationQueryPoint):
    id: str


class QueryVisualizationProjection(BaseModel):
    method: ProjectionMethod
    points: list[QueryVisualizationCoordinate]
    query: QueryVisualizationQueryPoint
    fallback: Literal["pca"] | None = None


class QueryVisualizationResponse(BaseModel):
    status: Literal["ready", "partial", "unavailable", "unsupported"]
    label: str
    technical_label: str
    note: str
    points: list[QueryVisualizationPoint]
    query: QueryVisualizationQueryPoint | None
    projection_method: ProjectionMethod = "pca"
    projections: list[QueryVisualizationProjection] = Field(default_factory=list)
    truncated: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    search: QuerySearchResponse
    visualization: QueryVisualizationResponse
    prompt: QueryPromptResponse


class DirectQueryResponse(BaseModel):
    question: str
    answer: str
    mode: Literal["generative"] = "generative"
    label: str = "Generative AI"
    description: str = (
        "Answer directly from the language model without retrieved documents."
    )
    retrieval_used: Literal[False] = False
    indexed_documents_used: Literal[False] = False
    prompt: QueryPromptResponse


class QueryModeResult(BaseModel):
    status: Literal["ready", "unavailable"]
    result: QueryResponse | None = None
    message: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "QueryModeResult":
        if (self.status == "ready") != (self.result is not None):
            raise ValueError("Ready query modes require a result")
        if self.status == "unavailable" and not self.message:
            raise ValueError("Unavailable query modes require a safe message")
        return self


class DirectModeResult(BaseModel):
    status: Literal["ready", "unavailable"]
    result: DirectQueryResponse | None = None
    message: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "DirectModeResult":
        if (self.status == "ready") != (self.result is not None):
            raise ValueError("Ready direct modes require a result")
        if self.status == "unavailable" and not self.message:
            raise ValueError("Unavailable direct modes require a safe message")
        return self


class QueryComparisonResults(BaseModel):
    semantic: QueryModeResult
    keyword: QueryModeResult
    hybrid: QueryModeResult
    generative: DirectModeResult


class AggregateQueryResponse(BaseModel):
    question: str
    results: QueryComparisonResults


class QueryErrorResponse(BaseModel):
    status: Literal["unavailable"] = "unavailable"
    message: str
