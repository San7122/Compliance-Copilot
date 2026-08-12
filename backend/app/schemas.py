from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str


class Citation(BaseModel):
    document: str
    section: str
    excerpt: str


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    answerable: bool


class HistoryItem(BaseModel):
    id: int
    question: str
    answer: str
    citations: list[Citation]
    confidence: str
    answerable: bool
    latency_ms: int | None = None
    created_at: str
