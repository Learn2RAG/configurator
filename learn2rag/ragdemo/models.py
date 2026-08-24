from typing import Literal

from pydantic import BaseModel, Field


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
