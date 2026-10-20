"""링크 가드레일 (말 쪽) — 전화로 URL·도메인을 읽어 주지 않는다. 링크는 send_app_link로 문자로만 (10/19 러프 버전).

보내는 쪽은 도구 인자가 안내 항목 id enum이라 형식 검증이 막는다 (catalog/notify.yaml, domain/guides.py).
여기서는 발화에 URL·도메인 모양이 나오면 등록 여부와 상관없이 그 지점부터 문장 끝까지 지운다 —
등록된 링크라도 읽어 주는 것 자체가 전화에 맞지 않는다. 다른 말로 바꾸지 않는다 (연락처 가드레일과 같은 원칙).
- 한계: 영문 URL 모양만 본다. 「토스 점 아이엠」처럼 한글로 읽은 주소는 못 잡는다.
"""

from __future__ import annotations

import re

_ENDS = ".?!。…"
_TLDS = "im|com|kr|net|org|io|me|ly|co"
URL = re.compile(
    rf"(?:https?://|www\.)[\w\-./?=&%#:~+]+"                                    # 스킴·www로 시작
    rf"|(?<![\w.@])[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:{_TLDS})\b(?:/[\w\-./?=&%#:~+]*)?",   # 도메인만 (toss.im/faq/139)
    re.IGNORECASE | re.ASCII)


def strip_links(text: str) -> tuple[str, list[str]]:
    """URL·도메인이 나온 지점부터 그 문장 끝까지 지운다 → (남은 말, 지운 URL들)."""
    removed: list[str] = []
    while m := URL.search(text):
        url = m.group(0).rstrip(".,")
        removed.append(url)
        end = m.start() + len(url)
        stop = next((i for i in range(end, len(text)) if text[i] in _ENDS), len(text) - 1)
        text = text[:m.start()].rstrip() + text[stop + 1:]
    return text, removed
