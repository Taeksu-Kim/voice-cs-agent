"""Document → Chunk (LangChain 스플리터 기반).

- 판매자 문서: MarkdownHeaderTextSplitter로 「## / ###」 단위 섹션을 나누고,
  섹션이 길면 RecursiveCharacterTextSplitter로 자른다. 섹션 경로("상위 > 하위")는 청크에 남긴다.
- FAQ: 질문 1건 = 1청크. 길면 RecursiveCharacterTextSplitter로 자른다.
- 후처리는 하나뿐: 표가 조각 경계에서 잘리면 뒤 조각에도 표 머리글을 넣는다.
"""

from __future__ import annotations

import re

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from csagent.knowledge.models import Chunk, Document

_H1 = re.compile(r"^#\s+.*\n?", re.M)
_HEADERS = [("##", "h2"), ("###", "h3")]
_TABLE_SEP = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


class Chunker:
    def __init__(self, max_chars: int = 1200) -> None:
        self._headers = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=True)
        # 자르는 우선순위: Q&A 경계(「Q3.」 줄 앞) → 문단 → 줄 → 공백 → 글자.
        # Q 경계를 문단보다 앞에 둬야 여러 문단짜리 답변이 질문과 떨어지지 않는다.
        self._length = RecursiveCharacterTextSplitter(
            chunk_size=max_chars, chunk_overlap=0, is_separator_regex=True,
            separators=[r"\n+(?=\s*\**\s*Q\d+[.)])", r"\n\n", r"\n", " ", ""],
        )

    def split(self, doc: Document) -> list[Chunk]:
        sections = [("", doc.body)] if doc.source == "faq" else self._sections(doc.body)
        pieces = [(path, part) for path, text in sections
                  for part in _repeat_table_header(self._length.split_text(text))]
        return [
            Chunk(chunk_id=f"{doc.doc_id}#{i}", doc_id=doc.doc_id, source=doc.source,
                  title=doc.title, section=path, text=part, url=doc.url, meta=dict(doc.meta))
            for i, (path, part) in enumerate(pieces)
        ]

    def split_all(self, docs: list[Document]) -> list[Chunk]:
        return [c for d in docs for c in self.split(d)]

    def _sections(self, body: str) -> list[tuple[str, str]]:
        """[(섹션 경로, 본문)] — 문서 제목(#)은 Document.title에 있으므로 떼고 나눈다."""
        out = []
        for d in self._headers.split_text(_H1.sub("", body, count=1)):
            path = " > ".join(d.metadata[k] for _, k in _HEADERS if d.metadata.get(k))
            # LangChain은 빈 줄을 지우고 그 자리를 "  \n"으로 남긴다 → 문단 경계로 복원
            text = d.page_content.replace("  \n", "\n\n").strip()
            if text:
                out.append((path, text))
        return out


def _repeat_table_header(pieces: list[str]) -> list[str]:
    """앞 조각 끝에서 이어지던 표가 다음 조각 첫 줄로 넘어가면, 그 표의 머리글 2줄을 앞에 붙인다."""
    out: list[str] = []
    header: list[str] = []  # 현재 이어지고 있는 표의 머리글 (머리글 줄 + 구분선)
    for piece in pieces:
        lines = piece.splitlines()
        starts_mid_table = (header and lines and lines[0].lstrip().startswith("|")
                            and not (len(lines) > 1 and _TABLE_SEP.match(lines[1].strip())))
        if starts_mid_table:
            piece = "\n".join(header + lines)
            lines = piece.splitlines()
        out.append(piece)
        header = _open_table_header(lines)
    return out


def _open_table_header(lines: list[str]) -> list[str]:
    """조각이 표 행으로 끝나면 그 표의 머리글 2줄, 아니면 []."""
    if not lines or not lines[-1].lstrip().startswith("|"):
        return []
    for i in range(len(lines) - 1, 0, -1):
        if not lines[i].lstrip().startswith("|"):
            break
        if _TABLE_SEP.match(lines[i].strip()):
            return [lines[i - 1], lines[i]]
    return []
