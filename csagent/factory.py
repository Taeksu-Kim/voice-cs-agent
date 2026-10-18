"""설정 → 실제 구성요소 조립. 실행 스크립트는 여기서 꺼내 쓰기만 한다."""

from __future__ import annotations

import logging
import os
from typing import Optional

from csagent.agent.baseline import BaselineAgent
from csagent.agent.controller import CallController
from csagent.agent.guardrails.contacts import ContactBook
from csagent.agent.guardrails.gate import ExecutionGate
from csagent.agent.loop import TurnLoop
from csagent.agent.router import Router
from csagent.agent.workers.specs import WORKERS
from csagent.agent.tools.registry import ToolRegistry
from csagent.config import Settings, load_settings
from csagent.llm import ChatLLM, OpenAICompatLLM
from csagent.llm_log import LLMCallLog, LoggedLLM
from csagent.obs import event
from csagent.retrieval.embedders import HttpDenseEmbedder, KiwiBM25
from csagent.retrieval.reranker import HttpReranker
from csagent.retrieval.retriever import HybridRetriever
from csagent.retrieval.store import MilvusStore


log = logging.getLogger("csagent.agent")

def build_retriever(with_reranker: bool = True, s: Optional[Settings] = None) -> HybridRetriever:
    s = s or load_settings()
    return HybridRetriever(
        MilvusStore(s.milvus_uri, s.collection, s.embed_dim),
        HttpDenseEmbedder(s.embedder_url, s.embedder_model, s.embed_dim),
        KiwiBM25().load(s.bm25_stats),
        HttpReranker(s.reranker_url, s.reranker_model) if with_reranker else None,
    )


def build_llm(s: Optional[Settings] = None, *, log: Optional[LLMCallLog] = None,
              source: str = "agent") -> ChatLLM:
    """log를 주면 호출마다 입출력을 기록한다 (csagent/llm_log.py)."""
    s = s or load_settings()
    llm = OpenAICompatLLM(s.llm_url, s.llm_model)
    return LoggedLLM(llm, log, source) if log else llm


def load_contacts(s: Optional[Settings] = None) -> Optional[ContactBook]:
    """scripts/extract_contacts.py 결과. 없으면 None (연락처 가드레일 꺼짐 — 경고만)."""
    s = s or load_settings()
    path = s.processed_dir / "contacts.jsonl"
    if not path.exists():
        event(log, "contacts_missing", logging.WARNING, path=str(path))
        return None
    return ContactBook.load(path, s.processed_dir / "chunks.jsonl")


def build_agent(kind: Optional[str] = None, s: Optional[Settings] = None, *,
                log: Optional[LLMCallLog] = None, tools: ToolRegistry):
    """kind: A0(단일·게이트 없음) | A1(단일·게이트) | C(컨트롤러+워커·게이트, 기본). 환경변수 AGENT로도 지정.
    tools: 도구 스택(csagent/stack.py)의 레지스트리 — 통화는 같은 스택의 open()으로 열어야 한다."""
    kind = (kind or os.environ.get("AGENT", "C")).upper().rstrip("'")
    retriever, llm = build_retriever(True, s), build_llm(s, log=log)
    if kind == "C":
        router_llm = llm.with_source("router") if isinstance(llm, LoggedLLM) else llm
        return CallController(TurnLoop(retriever, llm, tools, gate=ExecutionGate(), contacts=load_contacts(s)),
                              Router(router_llm, WORKERS), WORKERS)
    if kind in ("A0", "A1"):
        return BaselineAgent(retriever, llm, tools, gate=ExecutionGate() if kind == "A1" else None)
    raise ValueError(f"알 수 없는 에이전트 구성: {kind}")
