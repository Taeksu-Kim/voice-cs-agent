"""FastAPI 백엔드 — 상담 세션을 열고, 고객 발화마다 에이전트 응답을 SSE로 스트리밍한다.

    GET  /api/health                     구성요소 상태
    GET  /api/scenarios                  통화를 열 때 고를 고객 상황(시나리오) 목록
    POST /api/sessions                   {scenario_id} → 세션 생성
    GET  /api/sessions/{id}              통화 기록
    POST /api/sessions/{id}/turn         {utterance} → SSE: retrieval · delta · tool_call · tool_result · done · error
    /                                    웹 화면 (web/)

클라이언트가 스트림 도중 연결을 끊으면 생성기가 닫혀 LLM 요청도 끊긴다 (끼어들기 처리의 기반).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from csagent.agent.baseline import reply_to_dict
from csagent.agent.contracts import CallContext, TurnInput
from csagent.agent.interface import Agent
from csagent.config import ROOT
from csagent.evaluation.scenario import Scenario, load_scenarios
from csagent.server.sessions import SessionStore


class SessionRequest(BaseModel):
    scenario_id: str


class TurnRequest(BaseModel):
    utterance: str


def create_app(agent: Agent, scenarios: list[Scenario], *,
               web_dir: Optional[Path] = ROOT / "web") -> FastAPI:
    app = FastAPI(title="customer-service-agent")
    store = SessionStore()
    by_id = {s.id: s for s in scenarios}

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "agent": getattr(agent, "name", ""), "scenarios": len(scenarios)}

    @app.get("/api/scenarios")
    def list_scenarios() -> list[dict]:
        return [{"id": s.id, "task": s.task, "behavior": s.behavior,
                 "customer_name": s.context.customer.get("name", ""),
                 "goal": s.customer.get("goal", ""), "opening": s.customer.get("opening", ""),
                 "orders": s.context.customer.get("orders", [])} for s in scenarios]

    @app.post("/api/sessions")
    def create_session(req: SessionRequest) -> dict:
        sc = by_id.get(req.scenario_id)
        if sc is None:
            raise HTTPException(404, f"시나리오 없음: {req.scenario_id}")
        ctx = CallContext(direction=sc.context.direction, audience=sc.context.audience,
                          companies=sc.context.companies, customer=dict(sc.context.customer),
                          today=sc.context.today)
        s = store.create(ctx, sc.id)
        return {"session_id": s.id, "scenario_id": sc.id, "today": ctx.today,
                "customer": ctx.customer}

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        s = _session(store, session_id)
        return {"session_id": s.id, "scenario_id": s.scenario_id, "history": s.state.history,
                "disposition": s.disposition, "ended": s.ended,
                "authenticated": s.state.authenticated,
                "replies": [reply_to_dict(r) for r in s.replies],
                "trace": s.state.trace.to_dicts()}

    @app.post("/api/sessions/{session_id}/turn")
    def turn(session_id: str, req: TurnRequest) -> StreamingResponse:
        s = _session(store, session_id)
        if s.ended:
            raise HTTPException(409, "이미 종료된 통화입니다")
        if not req.utterance.strip():
            raise HTTPException(422, "발화가 비어 있습니다")

        def events() -> Iterator[str]:
            trace, sent, t0 = s.state.trace, len(s.state.trace.events), time.time()
            try:
                for ev in agent.respond_stream(s.state, TurnInput(req.utterance)):
                    # 에이전트 이벤트 사이사이에 새로 쌓인 추적 기록을 같이 내보낸다 (실시간 관찰용)
                    for e in trace.events[sent:]:
                        yield _sse("trace", _trace_view(e, t0))
                    sent = len(trace.events)
                    if ev.type == "done":
                        reply = ev.data["reply"]
                        s.record(reply)
                        yield _sse("done", {**reply_to_dict(reply), "session_ended": s.ended,
                                            "state": state_view(s.state)})
                    else:
                        yield _sse(ev.type, ev.data)
            except Exception as e:  # 스트림 중 오류는 이벤트로 알린다 (HTTP 상태는 이미 200)
                yield _sse("error", {"message": f"{type(e).__name__}: {e}"})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if web_dir and web_dir.exists():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


def create_default_app() -> FastAPI:
    """실제 구성요소로 조립한 앱 (uvicorn 진입점)."""
    from csagent.factory import build_agent

    return create_app(build_agent(), load_scenarios(ROOT / "scenarios"))


def _session(store: SessionStore, session_id: str):
    s = store.get(session_id)
    if s is None:
        raise HTTPException(404, "세션 없음")
    return s


def _trace_view(e, t0: float) -> dict:
    data = {k: v for k, v in e.data.items() if k != "texts"}   # 검색 본문은 화면에 보내지 않는다
    return {"kind": e.kind, "ms": round((e.t - t0) * 1000), "data": data}


def state_view(state) -> dict:
    """통화 상태 요약 — 화면의 「상태」 칸."""
    p = state.pending
    return {
        "turn": state.turn,
        "authenticated": state.authenticated, "auth_failures": state.auth_failures,
        "contact_verified": state.contact_verified,
        "pending": {"kind": p.kind, "action": p.action, "target": p.target, "turn": p.turn} if p else None,
        "confirmations": [{"action": c.action, "target": c.target, "turn": c.turn} for c in state.confirmations],
        "executed": sorted(state.executed),
        "active": {"task": state.active.name, "slots": state.active.slots} if state.active else None,
        "queue": [t.name for t in state.queue],
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
