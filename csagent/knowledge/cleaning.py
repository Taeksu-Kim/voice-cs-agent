"""원문 정제 헬퍼 (상태 없는 순수 함수만).

- GitBook Markdown: 상단 안내문·하단「Agent Instructions」지시문·이미지·템플릿 태그 제거.
  하단 지시문은 문서 내용이 아니라 AI에게 하는 지시라서, 지식베이스에 남으면
  검색 결과를 통해 에이전트 프롬프트로 흘러 들어간다 (인젝션 경로).
- FAQ: 「○○ 바로가기」링크 줄을 음성 안내에 쓸 수 있는 표식으로 바꾼다.
"""

from __future__ import annotations

import html
import re

_GITBOOK_TRAILER = re.compile(r"\n-{3,}\s*\n+#\s*Agent Instructions.*", re.S)
_GITBOOK_HEADER = re.compile(r"^>\s*For the complete documentation index.*?\n", re.M)
_FIGURE = re.compile(r"<figure>.*?</figure>", re.S)
_IMG = re.compile(r"<img[^>]*>")
_TEMPLATE_TAG = re.compile(r"\{%.*?%\}")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HARD_BREAK = re.compile(r"\\[ \t]*$", re.M)
_CELL_BREAK = re.compile(r"</li>\s*<li>|</p>\s*<p>|<br\s*/?>", re.I)
_CELL_END = re.compile(r"</t[dh]>", re.I)
_ROW_END = re.compile(r"</tr>|<br\s*/?>", re.I)
_MD_ESCAPE = re.compile(r"\\([\[\]()*_#>~`-])")
_LINK_LINE = re.compile(r"^\s*(?:[-*>]\s*)?(.+?)\s*바로가기\s*$")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_gitbook_noise(md: str) -> str:
    md = _GITBOOK_TRAILER.sub("", md)
    md = _GITBOOK_HEADER.sub("", md)
    md = _FIGURE.sub("", md)
    md = _IMG.sub("", md)
    md = _TEMPLATE_TAG.sub("", md)
    # Markdown 표 칸 안의 HTML 목록·줄바꿈: 행이 깨지지 않게 「 / 」로 이어 붙인다
    md = re.sub(r"^\|.*$", lambda m: _CELL_BREAK.sub(" / ", m.group(0)), md, flags=re.M)
    # HTML 표: 태그를 지우기 전에 칸·행 경계를 남긴다 (안 그러면 칸 내용이 붙어 버린다)
    md = _CELL_END.sub(" | ", md)
    md = _ROW_END.sub("\n", md)
    md = _HTML_TAG.sub("", md)
    # 링크는 글자만 남긴다 (판매자 문서 내부 경로는 상담 안내에 쓸 수 없음)
    md = _MD_LINK.sub(r"\1", _MD_LINK.sub(r"\1", md))  # 중첩 [[a](x)](y) 대비 두 번
    md = _HARD_BREAK.sub("", md)
    md = _MD_ESCAPE.sub(r"\1", html.unescape(md))
    # Markdown 표 정렬용 공백 (실측: 597자 행의 대부분이 공백) → 한 칸으로
    md = re.sub(r"^\|.*$", lambda m: re.sub(r"-{4,}", "---", re.sub(r" {2,}", " ", m.group(0))),
                md, flags=re.M)
    return normalize_whitespace(md)


def is_not_found_page(md: str) -> bool:
    return bool(re.search(r"^#\s*Page Not Found", md, re.M))


def mark_app_links(text: str) -> str:
    """「- 주문 내역 바로가기」→「[앱 링크: 주문 내역]」 (링크 발송 대상이라는 표식)."""
    lines = []
    for line in text.splitlines():
        m = _LINK_LINE.match(line)
        lines.append(f"[앱 링크: {m.group(1)}]" if m else line)
    return "\n".join(lines)
