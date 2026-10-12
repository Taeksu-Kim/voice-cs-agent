"""임베더 — Dense(벡터)와 Sparse(키워드 BM25).

외부 서버는 Protocol 뒤에 두고, 테스트는 Fake 구현으로 서버 없이 돌린다.
Sparse 벡터는 Milvus 형식인 {토큰 인덱스: 가중치} dict로 다룬다.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

SparseVec = dict[int, float]

# Qwen3-Embedding은 쿼리 쪽에만 지시문을 붙이는 방식을 권장한다 (문서 쪽은 원문).
DEFAULT_QUERY_INSTRUCTION = (
    "Given a customer's question to a fintech and e-commerce customer service, "
    "retrieve the help-center passages that answer it"
)


class DenseEmbedder(Protocol):
    dim: int

    def encode_docs(self, texts: Sequence[str]) -> list[list[float]]: ...

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


class SparseEmbedder(Protocol):
    def encode_docs(self, texts: Sequence[str]) -> list[SparseVec]: ...

    def encode_queries(self, texts: Sequence[str]) -> list[SparseVec]: ...


# ---------------------------------------------------------------- dense


class HttpDenseEmbedder:
    """OpenAI 호환 /embeddings 서버 (vLLM pooling 또는 CPU 임베딩 서버)."""

    def __init__(self, base_url: str, model: str, dim: int = 1024, *,
                 query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
                 batch: int = 64, max_chars: int = 3500, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.query_instruction = query_instruction
        self.batch = batch
        self.max_chars = max_chars  # 서버 max-model-len 초과 시 400 → 클라이언트에서 자른다
        self.timeout = timeout

    def encode_docs(self, texts: Sequence[str]) -> list[list[float]]:
        return self._post([t[:self.max_chars] for t in texts])

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        prefix = f"Instruct: {self.query_instruction}\nQuery: " if self.query_instruction else ""
        return self._post([prefix + t for t in texts])

    def _post(self, texts: list[str]) -> list[list[float]]:
        import requests  # lazy

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            r = requests.post(f"{self.base_url}/embeddings",
                              json={"model": self.model, "input": texts[i:i + self.batch]},
                              timeout=self.timeout)
            r.raise_for_status()
            out += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
        return out


class FakeDenseEmbedder:
    """텍스트 해시로 만든 결정적 단위벡터 (테스트 전용)."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def encode_docs(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    encode_queries = encode_docs

    def _vec(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).standard_normal(self.dim)
        return (v / (np.linalg.norm(v) or 1.0)).tolist()


# ---------------------------------------------------------------- sparse


KEEP_TAGS = {"NNG", "NNP", "NNB", "SN", "SL", "XR"}


def kiwi_tokenizer() -> Callable[[str], list[str]]:
    """kiwi 형태소 → 내용어만 (조사·어미 제거: 「휴가를」=「휴가가」)."""
    from kiwipiepy import Kiwi  # lazy

    kiwi = Kiwi()
    return lambda text: [t.form for t in kiwi.tokenize(text) if t.tag in KEEP_TAGS]


def _token_index(tok: str) -> int:
    return int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16) % (2**31)


class KiwiBM25:
    """형태소 BM25 (agentic_rag에서 이식 — 어절 BM25 대비 실측 개선).

    문서 쪽 = tf 정규화 가중치, 쿼리 쪽 = IDF → 내적이 곧 BM25 점수.
    IDF·평균 길이는 코퍼스로 fit하고 저장해 두었다가 쿼리 시 같은 통계를 쓴다.
    """

    def __init__(self, tokenizer: Optional[Callable[[str], list[str]]] = None,
                 k1: float = 1.5, b: float = 0.75) -> None:
        self._tokenize = tokenizer or kiwi_tokenizer()
        self.k1, self.b = k1, b
        self.idf: dict[str, float] = {}
        self.avgdl = 1.0

    def fit(self, corpus: Sequence[str]) -> "KiwiBM25":
        docs = [self._tokenize(t) for t in corpus]
        df: dict[str, int] = {}
        for toks in docs:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(docs)
        self.idf = {t: math.log((n - d + 0.5) / (d + 0.5) + 1.0) for t, d in df.items()}
        self.avgdl = sum(map(len, docs)) / n if n else 1.0
        return self

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"idf": self.idf, "avgdl": self.avgdl}, ensure_ascii=False))

    def load(self, path: Path) -> "KiwiBM25":
        d = json.loads(Path(path).read_text())
        self.idf, self.avgdl = d["idf"], d["avgdl"]
        return self

    def encode_docs(self, texts: Sequence[str]) -> list[SparseVec]:
        out = []
        for text in texts:
            toks = self._tokenize(text)
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            norm = self.k1 * (1 - self.b + self.b * (len(toks) or 1) / self.avgdl)
            out.append({_token_index(t): f * (self.k1 + 1) / (f + norm) for t, f in tf.items()})
        return out

    def encode_queries(self, texts: Sequence[str]) -> list[SparseVec]:
        # 코퍼스에 없는 토큰은 매칭 불가라 뺀다
        return [{_token_index(t): self.idf[t] for t in set(self._tokenize(text)) if t in self.idf}
                for text in texts]
