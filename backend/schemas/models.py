from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class Source(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    page: int | None = None
    section: str | None = None
    score: float | None = None
    text: str


class DocumentSummary(BaseModel):
    id: str
    filename: str
    checksum: str
    status: str
    chunk_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class PageResponse(BaseModel):
    items: list[DocumentSummary]
    total: int
    offset: int
    limit: int


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    top_k: int = Field(default=6, ge=1, le=20)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=10)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=4000)
    top_k: int = Field(default=10, ge=1, le=50)


class GraphQueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=4000)
    limit: int = Field(default=30, ge=1, le=100)


class GraphResponse(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
