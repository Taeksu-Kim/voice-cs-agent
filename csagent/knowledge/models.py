"""지식베이스 데이터 단위: 원문 문서(Document) → 검색 단위(Chunk)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str          # "faq:139", "shop:shopping-operations__shop__support"
    source: str          # "faq" | "shopping_doc"
    title: str
    body: str            # 정제된 본문
    url: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str        # "{doc_id}#{n}"
    doc_id: str
    source: str
    title: str
    section: str         # 문서 내 소제목 (FAQ는 빈 문자열)
    text: str
    url: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def embedding_text(self) -> str:
        """색인용 텍스트 — 제목·소제목을 앞에 붙여 짧은 본문도 맥락을 갖게 한다."""
        head = self.title if not self.section else f"{self.title} > {self.section}"
        return f"{head}\n{self.text}"
