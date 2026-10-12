"""검색 필터 → Milvus 필터식.

원천에서 사실로 주어진 값만 필터로 쓴다 (추정값은 노이즈가 될 수 있어 쓰지 않는다).
- audience · companies: 요청 경로로 확정되는 값
- 판매 종료 상품은 기본 제외
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SearchFilter:
    audience: Optional[str] = None            # "consumer" | "merchant" | "seller"
    companies: tuple[str, ...] = ()
    include_discontinued: bool = False

    def to_expr(self) -> str:
        parts = []
        if self.audience:
            parts.append(f"audience == {json.dumps(self.audience, ensure_ascii=False)}")
        if self.companies:
            parts.append(f"array_contains_any(companies, {_list(self.companies)})")
        if not self.include_discontinued:
            parts.append("discontinued == false")
        return " and ".join(parts)


def _list(values: tuple[str, ...]) -> str:
    return json.dumps(list(values), ensure_ascii=False)
