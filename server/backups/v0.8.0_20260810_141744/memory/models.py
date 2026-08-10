from dataclasses import dataclass
from typing import Optional


@dataclass
class Memory:
    id: int
    category: str
    key: str
    value: str
    importance: int
    created_at: str
    updated_at: str


@dataclass
class MemoryInput:
    category: str
    key: str
    value: str
    importance: int = 5


@dataclass
class MemorySearchResult:
    id: int
    category: str
    key: str
    value: str
    importance: int
    score: float
    created_at: str
    updated_at: str