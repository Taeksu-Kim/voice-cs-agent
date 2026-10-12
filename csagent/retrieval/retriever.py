"""하이브리드 검색기: 질의 → dense·sparse 각각 검색 → RRF 결합 → (리랭크) → 상위 k.

에이전트가 쓰는 검색 진입점은 이 클래스 하나다.
필터(SearchFilter)는 원천에서 사실로 주어진 값(상담 대상·계열사·판매종료)만 담는다.
"""

from __future__ import annotations

from typing import Literal, Optional

from csagent.retrieval.embedders import DenseEmbedder, SparseEmbedder
from csagent.retrieval.filters import SearchFilter
from csagent.retrieval.fusion import rrf
from csagent.retrieval.models import Hit
from csagent.retrieval.reranker import Reranker
from csagent.retrieval.store import MilvusStore

Mode = Literal["hybrid", "dense", "sparse"]


class HybridRetriever:
    def __init__(self, store: MilvusStore, dense: DenseEmbedder, sparse: SparseEmbedder,
                 reranker: Optional[Reranker] = None, *, prefetch: int = 30) -> None:
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.prefetch = prefetch

    def search(self, query: str, k: int = 5, *, flt: Optional[SearchFilter] = None,
               mode: Mode = "hybrid", rerank: bool = True) -> list[Hit]:
        """mode·rerank는 ablation용 스위치 (기본: hybrid + 리랭크)."""
        candidates = self._first_stage(query, (flt or SearchFilter()).to_expr(), mode)
        if rerank and self.reranker is not None and candidates:
            scores = self.reranker.rerank(query, [h.display_text() for h in candidates])
            candidates = sorted((h.with_score(s) for h, s in zip(candidates, scores)),
                                key=lambda h: h.score, reverse=True)
        return candidates[:k]

    def _first_stage(self, query: str, expr: str, mode: Mode) -> list[Hit]:
        rankings = []
        if mode in ("hybrid", "dense"):
            rankings.append(self.store.search_dense(
                self.dense.encode_queries([query])[0], self.prefetch, expr=expr))
        if mode in ("hybrid", "sparse"):
            rankings.append(self.store.search_sparse(
                self.sparse.encode_queries([query])[0], self.prefetch, expr=expr))
        return rankings[0] if len(rankings) == 1 else rrf(rankings)
