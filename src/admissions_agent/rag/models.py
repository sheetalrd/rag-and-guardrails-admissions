from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    text: str
    source: str
    page: int


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    source: str
    page: int
    distance: float
