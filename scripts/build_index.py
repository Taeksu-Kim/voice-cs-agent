"""지식베이스 전량 재색인: 원문 → 정제·청크 → BM25 fit → 임베딩 → Milvus.

필요: 임베딩 서버(EMBEDDER_URL). 청크는 data/processed/chunks.jsonl에도 남겨 눈으로 점검한다.

    python scripts/build_index.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.config import load_settings
from csagent.knowledge.chunking import Chunker
from csagent.knowledge.loaders import FaqLoader, ShoppingDocLoader
from csagent.retrieval.embedders import HttpDenseEmbedder, KiwiBM25
from csagent.retrieval.store import MilvusStore


def main() -> None:
    s = load_settings()
    t0 = time.time()

    docs = (FaqLoader(s.raw_dir / "toss_faq.jsonl").load()
            + ShoppingDocLoader(s.raw_dir / "shopping_docs").load())
    chunks = Chunker().split_all(docs)
    texts = [c.embedding_text() for c in chunks]
    print(f"문서 {len(docs)} → 청크 {len(chunks)}")

    s.processed_dir.mkdir(parents=True, exist_ok=True)
    with (s.processed_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    bm25 = KiwiBM25().fit(texts)
    bm25.save(s.bm25_stats)
    sparse = bm25.encode_docs(texts)
    print(f"BM25 fit·인코딩 {time.time() - t0:.1f}s (어휘 {len(bm25.idf)})")

    dense = HttpDenseEmbedder(s.embedder_url, s.embedder_model, s.embed_dim).encode_docs(texts)
    print(f"dense 인코딩 {time.time() - t0:.1f}s")

    store = MilvusStore(s.milvus_uri, s.collection, s.embed_dim)
    store.recreate()
    store.insert(chunks, dense, sparse)
    print(f"Milvus '{s.collection}' 적재 {store.count()}건 — 총 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
