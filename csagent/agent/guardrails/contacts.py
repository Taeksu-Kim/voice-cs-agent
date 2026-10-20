"""연락처 가드레일 — 에이전트가 말하는 전화번호는 이번 통화에서 검색된 문서의 연락처에 있는 것만.

연락처는 scripts/extract_contacts.py가 청크마다 LLM으로 뽑은 것 (누구 번호인지·어떤 경우에 쓰는지).
- 입력 쪽: [참고 문서]의 문서마다 [연락처] 줄을 붙여, 본문 속 번호가 누구 것이고 언제 쓰는지 알려 준다
- 출력 쪽: 발화에 전화번호가 나오면 목록과 대조해, 없는 번호는 지운다 (「토스뱅크 고객센터(1661-7655)로」 →
  「토스뱅크 고객센터로」). 문장을 다른 말로 바꾸지 않는다 — 대체 문장이 또 다른 약속을 지어낼 수 있다.
  스트리밍에서는 숫자가 나온 지점부터 그 문장 끝까지만 붙잡았다가 검사 후 내보낸다 (앞부분은 바로 나감).
- 한계: 8자리 이상 전화번호 모양만 본다. 1332(금감원)·126(국세청) 같은 짧은 대표번호는 금액·카드 끝자리와
  구분이 안 돼 검사하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

from csagent.agent.guardrails.links import strip_links
from csagent.obs import event

log = logging.getLogger("csagent.agent")

# 대표번호 1XXX-XXXX · 지역/휴대폰 0XX-XXX(X)-XXXX
PHONE = re.compile(r"(?<![\d-])(?:1\d{3}[-\s.]?\d{4}|0\d{1,2}[-\s.]?\d{3,4}[-\s.]?\d{4})(?!\d)")
_ENDS = ".?!。…"


def digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def text_hash(title: str, text: str) -> str:
    """청크가 바뀌었는지 알아보는 값 — 추출 당시 청크와 지금 청크가 다르면 그 연락처는 쓰지 않는다."""
    return hashlib.sha1((title + "\n" + text).encode("utf-8")).hexdigest()[:12]


class ContactBook:
    def __init__(self, by_chunk: dict[str, list[dict]]) -> None:
        # 전화만 (팩스는 안내 대상 아님)
        self.by_chunk = {cid: [c for c in cs if c.get("kind", "전화") == "전화"] for cid, cs in by_chunk.items()}

    @classmethod
    def load(cls, contacts_path: Path, chunks_path: Path) -> "ContactBook":
        current = {}
        for line in Path(chunks_path).read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            current[c["chunk_id"]] = text_hash(c["title"], c["text"])
        by_chunk, stale = {}, []
        for line in Path(contacts_path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if current.get(r["chunk_id"]) == r["hash"]:
                by_chunk[r["chunk_id"]] = r["contacts"]
            else:
                stale.append(r["chunk_id"])
        if stale:   # 재색인으로 청크가 바뀌었다 → extract_contacts.py를 다시 돌려야 함
            event(log, "contacts_stale", logging.WARNING, count=len(stale), sample=stale[:5])
        return cls(by_chunk)

    def of(self, chunk_id: str, toss_only: bool = False) -> list[dict]:
        """toss_only: 토스 계열 확인 번호(TOSS_CONTACTS 표에 있는 것)만 — 카드사·은행 번호는 빠진다."""
        return [c for c in self.by_chunk.get(chunk_id, []) if c.get("verified") or not toss_only]

    def allowed(self, chunk_ids: Iterable[str], toss_only: bool = False) -> set[str]:
        return {digits(c["phone"]) for cid in chunk_ids for c in self.of(cid, toss_only)}

    def render(self, chunk_id: str, toss_only: bool = False) -> str:
        """[참고 문서] 문서 밑에 붙일 줄. 토스 계열 번호는 원문 확인 표의 주인으로 (LLM이 붙인 이름보다 우선)."""
        lines = []
        for c in self.of(chunk_id, toss_only):
            org, svc = c.get("org", ""), c.get("service", "")
            if c.get("verified") and svc not in org.replace(" ", ""):
                org = f"{svc} 고객센터"
            line = f"[연락처] {org or '기관 미상'} {c['phone']}"
            if c.get("when"):
                line += f" — {c['when']}"
            if c.get("hours"):
                line += f" ({c['hours']})"
            lines.append(line)
        return "\n".join(lines)


def strip_unlisted(text: str, allowed: Optional[set[str]], before: str = "") -> tuple[str, list[str]]:
    """목록에 없는 전화번호를 지운다. allowed=None이면 검사하지 않음 (연락처 목록이 없는 구성).
    before: text 앞에 이미 나간 말 (스트리밍 조각) — 조사 「으로/로」를 고를 때 본다."""
    if allowed is None:
        return text, []
    removed = [m.group(0) for m in PHONE.finditer(text) if digits(m.group(0)) not in allowed]
    for num in removed:
        m = re.search(r"\s?" + re.escape(num) + r"(으로)?", text)
        tail = m.group(1) or ""
        if tail and _no_final(before + text[:m.start()]):
            tail = "로"     # 「고객센터 1577-6000으로」 → 「고객센터로」 (받침 없는 말 뒤 「으로」는 「로」)
        text = text[:m.start()] + tail + text[m.end():]
    if removed:
        text = re.sub(r"\(\s*\)", "", text)        # 「고객센터()로」 → 「고객센터로」
        text = re.sub(r"\s{2,}", " ", text)   # 앞뒤 공백은 그대로 — 스트리밍 조각의 앞 공백일 수 있다
    return text, removed


def mask_unlisted(text: str, allowed: set[str]) -> str:
    """[참고 문서] 본문에서 말하면 안 되는 번호를 가린다 — 모델이 보지 못하면 말하려다 지워져 문장이 깨지는 일
    (「현대카드 고객센터 번호는입니다.」, runs/20261016_182034_C)도 없다. 번호를 좁힌 업무(toss_numbers_only)에서만 쓴다."""
    return PHONE.sub(lambda m: m.group(0) if digits(m.group(0)) in allowed else "(번호 생략)", text)


def _no_final(before: str) -> bool:
    """앞말 마지막 글자가 받침 없는 한글이거나 ㄹ 받침인가 (「으로」 대신 「로」를 쓰는 경우)."""
    if not before or not "가" <= before[-1] <= "힣":
        return False
    final = (ord(before[-1]) - 0xAC00) % 28
    return final in (0, 8)


def _latin(ch: str) -> bool:
    return "a" <= ch.lower() <= "z"


class SpeechScreen:
    """스트리밍 조각용. 숫자·영문이 없는 동안은 그대로 내보내고, 나오면 그 문장 끝까지 모았다가 검사한다
    (전화번호 → strip_unlisted, URL·도메인 → links.strip_links). allowed=None이면 번호 검사만 빠진다."""

    def __init__(self, allowed: Optional[set[str]]) -> None:
        self.allowed = allowed
        self.held = ""
        self.dot = False       # 「영문·숫자.」 바로 뒤 — 문장 끝인지 URL·소수점 가운데인지 다음 글자를 보고 안다
        self.gap = ""          # 숫자 바로 앞의 공백·여는 괄호 — 번호를 지울 때 같이 지워야 해서 잠깐 쥐고 있는다
        self.removed: list[str] = []
        self.links: list[str] = []
        self.last = ""         # 마지막으로 내보낸 글자 — 번호를 지운 뒤 조사를 고를 때

    def feed(self, piece: str) -> str:
        out = ""
        for ch in piece:
            if self.held and self.dot:
                self.dot = False
                if ch.isspace():
                    out += self._release()   # 앞 마침표가 문장 끝이었다 — 이 공백은 아래에서 평소대로
            if self.held:
                self.held += ch
                if ch == "." and self.held[-2:-1].isascii() and self.held[-2:-1].isalnum():
                    self.dot = True          # toss.im · 1.5 — 아직 놓지 않는다
                elif ch in _ENDS:
                    out += self._release()
            elif ch.isdigit() or _latin(ch):
                self.held, self.gap = self.gap + ch, ""
            elif ch in " (":
                self.gap += ch
            else:
                out += self.gap + ch
                self.gap, self.last = "", ch
        return out

    def flush(self) -> str:
        out = self._release() if self.held else self.gap
        self.gap, self.dot = "", False
        return out

    def _release(self) -> str:
        text, removed = strip_unlisted(self.held, self.allowed, before=self.last)
        if not removed:
            text = self.held   # 그대로 (strip_unlisted는 지운 게 있을 때만 공백을 정리한다)
        # 지웠으면 번호 앞 공백·괄호도 같이 사라진다 (「고객센터 1661-7655로」 → 「고객센터로」)
        text, links = strip_links(text)
        self.removed += removed
        self.links += links
        self.held = ""
        self.last = (self.last + text)[-1:]
        return text
