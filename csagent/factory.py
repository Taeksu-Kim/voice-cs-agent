"""설정 → 실제 구성요소 조립. 실행 스크립트는 여기서 꺼내 쓰기만 한다."""

from __future__ import annotations

import os
from typing import Optional

from csagent.agent.baseline import BaselineAgent
from csagent.agent.guardrails.gate import ExecutionGate
from csagent.agent.tools.mock import default_registry
from csagent.config import Settings, load_settings
from csagent.llm import OpenAICompatLLM
from csagent.retrieval.embedders import HttpDenseEmbedder, KiwiBM25
from csagent.retrieval.reranker import HttpReranker
from csagent.retrieval.retriever import HybridRetriever
from csagent.retrieval.store import MilvusStore


def build_retriever(with_reranker: bool = True, s: Optional[Settings] = None) -> HybridRetriever:
    s = s or load_settings()
    return HybridRetriever(
        MilvusStore(s.milvus_uri, s.collection, s.embed_dim),
        HttpDenseEmbedder(s.embedder_url, s.embedder_model, s.embed_dim),
        KiwiBM25().load(s.bm25_stats),
        HttpReranker(s.reranker_url, s.reranker_model) if with_reranker else None,
    )


def build_llm(s: Optional[Settings] = None) -> OpenAICompatLLM:
    s = s or load_settings()
    return OpenAICompatLLM(s.llm_url, s.llm_model)


def build_agent(kind: Optional[str] = None, s: Optional[Settings] = None):
    """kind: A0(게이트 없음) | A1(게이트 있음, 기본). C'는 구현 후 추가. 환경변수 AGENT로도 지정."""
    kind = (kind or os.environ.get("AGENT", "A1")).upper()
    if kind not in ("A0", "A1"):
        raise ValueError(f"알 수 없는 에이전트 구성: {kind}")
    return BaselineAgent(build_retriever(True, s), build_llm(s), default_registry(),
                         gate=ExecutionGate() if kind == "A1" else None)
