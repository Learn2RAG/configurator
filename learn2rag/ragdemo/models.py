from typing import Literal

from pydantic import BaseModel, Field, field_validator


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


class QuerySearchResponse(BaseModel):
    mode: str
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


class QueryResponse(BaseModel):
    question: str
    answer: str
    search: QuerySearchResponse
    prompt: QueryPromptResponse


class QueryErrorResponse(BaseModel):
    status: Literal["unavailable"] = "unavailable"
    message: str
