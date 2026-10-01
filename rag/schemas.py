from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel


@dataclass
class RawDoc:
    text: str
    source: str            # path or URL, shown in citations
    doc_type: str          # "code" | "doc" | "pdf"
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    doc_type: str
    metadata: dict = field(default_factory=dict)


class Citation(BaseModel):
    chunk_id: str
    source: str


class RAGAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]


class EvalPair(BaseModel):
    question: str
    answer: str
    source_chunk_id: str
