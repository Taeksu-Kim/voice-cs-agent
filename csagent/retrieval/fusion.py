"""검색 결과 결합 헬퍼.

dense·sparse 점수는 척도가 달라 그대로 더할 수 없다 → 순위 기반 RRF가 기본.
"""

from __future__ import annotations

from typing import Sequence

from csagent.retrieval.models import Hit


def rrf(rankings: Sequence[Sequence[Hit]], *, k: int = 60,
        weights: Sequence[float] | None = None) -> list[Hit]:
    """Reciprocal Rank Fusion: score = Σ w / (k + rank). 같은 chunk_id는 하나로 합친다."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    first: dict[str, Hit] = {}
    for w, ranking in zip(weights, rankings, strict=True):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + w / (k + rank)
            first.setdefault(hit.chunk_id, hit)
    return sorted((first[cid].with_score(s) for cid, s in scores.items()),
                  key=lambda h: h.score, reverse=True)
