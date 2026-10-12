"""검색 결과 단위."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class Hit:
    chunk_id: str
    score: float
    doc_id: str = ""
    source: str = ""
    title: str = ""
    section: str = ""
    text: str = ""
    url: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def with_score(self, score: float) -> "Hit":
        return replace(self, score=score)

    def display_text(self) -> str:
        """리랭커·에이전트가 읽는 텍스트 (색인 텍스트와 같은 형식)."""
        head = self.title if not self.section else f"{self.title} > {self.section}"
        return f"{head}\n{self.text}"
