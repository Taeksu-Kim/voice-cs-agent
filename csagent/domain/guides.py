"""안내 항목 — 항목 id → {이름, 대상, doc_id, url}. url은 지식 베이스(chunks.jsonl)의 값만 쓴다.

send_app_link는 항목 id만 받는다 (카탈로그 enum → inputSchema → 에이전트 형식 검증). 여기는 그 뒷단:
목록에 없는 id는 UNKNOWN_GUIDE로 거절하고, 보낸 url은 지식 베이스에서 찾은 것이다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from csagent.config import ROOT

ITEMS = Path(__file__).with_name("guides.yaml")
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"


@dataclass(frozen=True)
class Guide:
    id: str
    name: str
    audience: str      # consumer | seller
    doc_id: str
    url: str


class GuideBook:
    def __init__(self, guides: list[Guide]) -> None:
        self.by_id = {g.id: g for g in guides}

    @classmethod
    def load(cls, items_path: Path = ITEMS, chunks_path: Path = CHUNKS) -> "GuideBook":
        urls: dict[str, str] = {}
        for line in Path(chunks_path).read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            urls.setdefault(c["doc_id"], c.get("url") or "")
        guides, missing = [], []
        for raw in yaml.safe_load(Path(items_path).read_text(encoding="utf-8")) or []:
            url = urls.get(raw["doc_id"])
            if not url:
                missing.append(f"{raw['id']}({raw['doc_id']})")
                continue
            guides.append(Guide(raw["id"], raw["name"], raw["audience"], raw["doc_id"], url))
        if missing:   # 재색인으로 문서가 빠졌다 — 지어낸 url로 대신하지 않고 시작을 멈춘다
            raise ValueError(f"지식 베이스에 없는 안내 항목: {missing}")
        ids = [g.id for g in guides]
        if len(ids) != len(set(ids)):
            raise ValueError(f"안내 항목 id 중복: {sorted({i for i in ids if ids.count(i) > 1})}")
        return cls(guides)

    def get(self, guide_id: str) -> Optional[Guide]:
        return self.by_id.get(guide_id)

    def urls(self) -> set[str]:
        return {g.url for g in self.by_id.values()}
