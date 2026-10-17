"""원천 데이터 → Document. 로더마다 출처 하나만 책임진다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from csagent.knowledge.cleaning import (
    is_not_found_page, mark_app_links, normalize_whitespace, strip_gitbook_noise,
)
from csagent.knowledge.models import Document
from csagent.knowledge.taxonomy import contacts, faq_audience, faq_domains, faq_service, shop_domains


class FaqLoader:
    """토스 고객센터 FAQ (scripts/collect_toss.py 산출물 toss_faq.jsonl)."""

    def __init__(self, path: Path, companies: Optional[Iterable[str]] = None) -> None:
        self.path = Path(path)
        self.companies = set(companies) if companies else None

    def load(self) -> list[Document]:
        # 같은 FAQ가 여러 계열사 워크스페이스에 중복 게시된다 (실측: 1,750행 중 고유 1,250).
        # id로 합치고, 게시된 계열사는 목록으로 남긴다.
        rows: dict[int, dict] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if self.companies and r["company"] not in self.companies:
                continue
            rows.setdefault(r["id"], {**r, "companies": []})["companies"].append(r["company"])
        return [self._to_document(r) for r in rows.values()]

    @staticmethod
    def _to_document(r: dict) -> Document:
        audience = faq_audience(r["companies"], r["title"])
        domains = faq_domains(r["category_ids"], r["title"], audience)
        body = normalize_whitespace(mark_app_links(r["answer"]))
        return Document(
            doc_id=f"faq:{r['id']}",
            source="faq",
            title=r["title"],
            body=body,
            url=f"https://support.toss.im/faq/{r['id']}",
            meta={
                "audience": audience,
                "companies": r["companies"],
                "domains": domains,
                "service": faq_service(r["companies"], r["title"], audience, domains, body),
                "contacts": contacts(body),
                # 「(2026.02.16 기준 판매종료)」 — 끝난 상품은 안내 대상에서 빼야 한다
                "discontinued": "판매종료" in r["title"],
                "view_count": r["view_count"],
                "updated": r["updated"],
            },
        )


class ShoppingDocLoader:
    """토스쇼핑 판매자·광고주 문서 (GitBook Markdown + _index.json)."""

    # dev·connect = 개발자용 API 명세 (상담 지식 아님)
    def __init__(self, directory: Path, exclude_sections: Iterable[str] = ("dev", "connect"),
                 min_body_chars: int = 80) -> None:
        self.dir = Path(directory)
        self.exclude = set(exclude_sections)
        self.min_body_chars = min_body_chars  # 「자주 묻는 질문을 모았어요」같은 목차 페이지 제외

    def load(self) -> list[Document]:
        index = json.loads((self.dir / "_index.json").read_text(encoding="utf-8"))
        docs = []
        for item in index:
            if "file" not in item or item["section"] in self.exclude:
                continue
            raw = (self.dir / item["file"]).read_text(encoding="utf-8")
            if is_not_found_page(raw):
                continue
            body = strip_gitbook_noise(raw)
            if len(body) - len(item["title"]) < self.min_body_chars:
                continue
            docs.append(Document(
                doc_id=f"shop:{item['file'].removesuffix('.md')}",
                source="shopping_doc",
                title=strip_gitbook_noise(item["title"]),  # 「\[포인트피드]」 같은 이스케이프 제거
                body=body,
                url=item["url"],
                meta={
                    "audience": "seller",
                    "companies": ["토스"],  # 토스쇼핑은 토스(비바리퍼블리카) 서비스
                    "domains": shop_domains(item["file"].removesuffix(".md")),
                    "service": "토스쇼핑",
                    "contacts": contacts(body),
                    "discontinued": False,
                },
            ))
        return docs
