"""Milvus 저장소 — 청크 1개 = 엔티티 1개 (dense + sparse 벡터를 함께 저장).

검색은 dense / sparse를 **따로** 호출한다. 점수 결합은 앱 쪽(fusion.py)에서 한다:
토스 검색 플랫폼처럼 키워드(Elasticsearch)·벡터(Milvus)가 다른 엔진이어도
결합 로직이 그대로 유지되게 하기 위함.

Milvus Lite(로컬 파일)와 Milvus Standalone 모두 uri만 다르다.
"""

from __future__ import annotations

from typing import Optional, Sequence

from csagent.knowledge.models import Chunk
from csagent.retrieval.embedders import SparseVec
from csagent.retrieval.models import Hit

_FIELDS = ["doc_id", "source", "title", "section", "text", "url", "meta"]


class MilvusStore:
    def __init__(self, uri: str, collection: str, dim: int) -> None:
        import logging

        from pymilvus import MilvusClient  # lazy

        # Milvus Lite가 미구현 gRPC 메서드(AllocTimestamp) 호출마다 남기는 오류 로그 — 동작과 무관
        logging.getLogger("grpc._server").setLevel(logging.CRITICAL)
        self.client = MilvusClient(uri)
        self.collection = collection
        self.dim = dim
        if self.client.has_collection(collection):
            # 새 클라이언트에서는 컬렉션이 released 상태 — 검색 전에 메모리에 올려야 한다
            self.client.load_collection(collection)

    # -- 적재 --

    def recreate(self) -> None:
        """컬렉션을 지우고 새로 만든다 (전량 재색인)."""
        from pymilvus import DataType

        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        schema = self.client.create_schema(auto_id=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
        schema.add_field("source", DataType.VARCHAR, max_length=32)
        schema.add_field("title", DataType.VARCHAR, max_length=1024)
        schema.add_field("section", DataType.VARCHAR, max_length=1024)
        schema.add_field("text", DataType.VARCHAR, max_length=16384)
        schema.add_field("url", DataType.VARCHAR, max_length=1024)
        schema.add_field("meta", DataType.JSON)
        # 필터용 스칼라 필드 (meta에도 같은 값이 있지만, 필터는 스칼라 필드로 거는 게 빠르고 명확하다)
        array = dict(element_type=DataType.VARCHAR, max_capacity=8, max_length=32)
        schema.add_field("audience", DataType.VARCHAR, max_length=16)
        schema.add_field("companies", DataType.ARRAY, **array)
        schema.add_field("discontinued", DataType.BOOL)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

        index = self.client.prepare_index_params()
        index.add_index("dense", index_type="AUTOINDEX", metric_type="COSINE")
        index.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        self.client.create_collection(self.collection, schema=schema, index_params=index)

    def insert(self, chunks: Sequence[Chunk], dense: Sequence[list[float]],
               sparse: Sequence[SparseVec], *, batch: int = 256) -> int:
        rows = [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "source": c.source, "title": c.title,
             "section": c.section, "text": c.text, "url": c.url, "meta": c.meta,
             "audience": c.meta.get("audience", ""),
             "companies": c.meta.get("companies", []),
             "discontinued": bool(c.meta.get("discontinued", False)),
             "dense": list(d), "sparse": s or {0: 0.0}}
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        for i in range(0, len(rows), batch):
            self.client.insert(self.collection, rows[i:i + batch])
        return len(rows)

    def count(self) -> int:
        res = self.client.query(self.collection, filter="", output_fields=["count(*)"])
        return int(res[0]["count(*)"])

    # -- 검색 --

    def search_dense(self, vec: Sequence[float], k: int, *, expr: str = "") -> list[Hit]:
        return self._search([list(vec)], "dense", k, expr)

    def search_sparse(self, vec: SparseVec, k: int, *, expr: str = "") -> list[Hit]:
        if not vec:  # 코퍼스에 없는 단어뿐인 질의
            return []
        return self._search([vec], "sparse", k, expr)

    def _search(self, data: list, field: str, k: int, expr: str) -> list[Hit]:
        res = self.client.search(self.collection, data=data, anns_field=field, limit=k,
                                 filter=expr, output_fields=_FIELDS)
        return [self._to_hit(r) for r in res[0]]

    @staticmethod
    def _to_hit(r: dict) -> Hit:
        e = r["entity"]
        return Hit(chunk_id=r["chunk_id"], score=float(r["distance"]),
                   **{f: e.get(f) or ({} if f == "meta" else "") for f in _FIELDS})
