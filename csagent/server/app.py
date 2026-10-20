"""FastAPI 백엔드 — 상담 세션을 열고, 고객 발화마다 에이전트 응답을 SSE로 스트리밍한다.

    GET  /api/health                     구성요소 상태
    GET  /api/scenarios                  시나리오 목록 (평가 시나리오 + 발신자 — 녹화 재생에서 고객 정보 찾기용)
    GET  /api/callers                    실제 테스트 탭의 발신자 목록과 상황 카드 (scenarios_live/, csagent/server/callers.py)
    POST /api/sessions                   {scenario_id, mode, situation} → 세션 생성 (scenario_id는 발신자 id도 됨)
    GET  /api/sessions/{id}              통화 기록
    POST /api/sessions/{id}/turn         {utterance} → SSE: retrieval · delta · tool_call · tool_result · done · error
    WS   /api/sessions/{id}/voice        실시간 음성 (mode=voice 세션): PCM 프레임 → VAD·STT·에이전트·TTS (csagent/voice)
    GET  /api/llm_calls/{id}?source=     LLM 호출 하나의 입력·출력 원문 (trace의 call_id)
    GET  /api/recordings?source=         데모 탭 재생 원본 목록 (logs/* 통화, runs/* 평가) / GET /api/recordings/{id}
    POST /api/tts                        {text, voice} → wav (데모 재생용, 음성 서버 대리)
    GET · POST · DELETE /api/demos       데모 시나리오 탭 = 골라 둔 녹화 목록 (demos/demos.json) — 응답은 기록 그대로
    GET  /api/monitor/sources            모니터링할 수 있는 로그 폴더 (runs/* 평가, logs/* 서버)
    GET  /api/monitor?source=            로그 폴더의 지표 (csagent/monitoring.py)
    /                                    웹 화면 (web/) — 상담 콘솔(index.html), 모니터(monitor.html)

로그: 서버 실행 날짜의 logs/<날짜>/ 에 events.jsonl · llm_calls.jsonl · calls.jsonl(끝난 통화의 trace).

클라이언트가 스트림 도중 연결을 끊으면 생성기가 닫혀 LLM 요청도 끊긴다 (끼어들기 처리의 기반).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from csagent.agent.baseline import reply_to_dict
from csagent.agent.contracts import CallContext, TurnInput
from csagent.agent.interface import Agent
from csagent.config import ROOT
from csagent.evaluation.scenario import Scenario, load_scenarios
from csagent.monitoring import aggregate, find_llm_call, list_sources
from csagent.obs import CALLS_FILE, event
from csagent.server.callers import Caller
from csagent.server.recordings import greeting_of, list_recordings, load_recording
from csagent.server.sessions import Session, SessionStore

log = logging.getLogger("csagent.server")
_SOURCE = re.compile(r"^(runs|logs)/[\w\-]+$")
DEMOS_FILE = "demos/demos.json"


class SessionRequest(BaseModel):
    scenario_id: str
    mode: Literal["text", "voice"] = "text"
    situation: str = ""


class DemoRequest(BaseModel):
    title: str
    desc: str = ""
    source: str
    record_id: str


class TurnRequest(BaseModel):
    utterance: str


class TTSRequest(BaseModel):
    text: str
    voice: str = "F1"


def create_app(agent: Agent, scenarios: list[Scenario], *, callers: list[Caller] = (),
               web_dir: Optional[Path] = ROOT / "web", log_dir: Optional[Path] = None,
               root: Path = ROOT, voice_url: Optional[str] = None,
               make_voice: Optional[Callable] = None, stack: Any = None, silence_s: float = 8.0,
               version: Optional[dict] = None) -> FastAPI:
    """log_dir: 이 서버의 로그 폴더 (끝난 통화 저장·LLM 원문 조회 기본값). None이면 통화를 파일에 남기지 않는다.
    callers: 실제 테스트 탭 발신자 — 세션은 시나리오처럼 열린다 (Caller.to_scenario).
    voice_url: scripts/voice_server.py 주소 — 없으면 음성 통화 불가. make_voice는 테스트용 (VoiceClient·Endpointer 대체).
    stack: 도구 스택(csagent/stack.py) — 세션은 stack.open으로 열린다 (업무 데이터를 도메인 API에 심음).
    silence_s: 음성 통화 무응답 대기 — 이만큼 아무도 말하지 않으면 「들리세요?」, 한 번 더면 종료 안내 후 끊는다 (docs/09 C).
    version: 코드·카탈로그 버전 (stack.record_versions) — 끝난 통화 기록에 붙인다."""
    app = FastAPI(title="customer-service-agent")
    store = SessionStore(stack.open) if stack is not None else SessionStore()
    scenarios = [c.to_scenario() for c in callers] + list(scenarios)
    by_id = {s.id: s for s in scenarios}
    callers_by_id = {c.id: c for c in callers}
    demos_lock = threading.Lock()
    calls_lock = threading.Lock()

    def save_call(s: Session, end_reason: str = "agent_ended") -> None:
        if s.saved:
            return
        s.saved = True
        event(log, "call_end", call_id=s.state.call_id, scenario=s.scenario_id, disposition=s.disposition,
              end_reason=end_reason, turns=len(s.replies), mode=s.mode)
        if log_dir is None:
            return
        record = {"call_id": s.state.call_id, "version": version, "scenario": s.scenario_id, "mode": s.mode,
                  "situation": s.situation,
                  "created_at": s.created_at, "end_reason": end_reason,
                  "disposition": s.disposition, "history": s.state.history,
                  "replies": [reply_to_dict(r) for r in s.replies], "trace": s.state.trace.to_dicts()}
        if s.voice_turns:
            record["voice_turns"] = s.voice_turns
        with calls_lock, (log_dir / CALLS_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def turn_events(s: Session, utterance: str,
                    cancel: Optional[threading.Event] = None) -> Iterator[tuple[str, dict]]:
        """한 턴의 에이전트 이벤트 (텍스트 SSE·음성 WebSocket 공용). cancel이 서면 생성기를 닫는다 — 그 턴은 기록 안 됨."""
        trace, sent, t0 = s.state.trace, len(s.state.trace.events), time.time()
        gen = agent.respond_stream(s.state, TurnInput(utterance))
        try:
            for ev in gen:
                if cancel is not None and cancel.is_set():
                    event(log, "turn_cancelled", call_id=s.state.call_id, turn=s.state.turn)
                    return
                # 에이전트 이벤트 사이사이에 새로 쌓인 추적 기록을 같이 내보낸다 (실시간 관찰용)
                for e in trace.events[sent:]:
                    yield "trace", _trace_view(e, t0)
                sent = len(trace.events)
                if ev.type == "done":
                    reply = ev.data["reply"]
                    s.record(reply)
                    if s.ended and s.mode != "voice":   # 음성은 TTS까지 끝난 뒤 VoiceCall이 저장 (voice_turns 포함)
                        save_call(s)
                    yield "done", {**reply_to_dict(reply), "session_ended": s.ended,
                                   "state": state_view(s.state)}
                else:
                    yield ev.type, ev.data
        except Exception as e:  # 스트림 중 오류는 이벤트로 알린다 (HTTP 상태는 이미 200)
            log.error("turn_error", exc_info=True, extra={"event": "turn_error", "fields": {
                "call_id": s.state.call_id, "turn": s.state.turn, "error": f"{type(e).__name__}: {e}"}})
            yield "error", {"message": f"{type(e).__name__}: {e}"}
        finally:
            gen.close()

    def source_dir(source: Optional[str]) -> Path:
        if not source:
            if log_dir is None:
                raise HTTPException(404, "서버 로그 폴더가 없습니다")
            return log_dir
        if not _SOURCE.match(source) or not (root / source).is_dir():
            raise HTTPException(404, f"로그 폴더 없음: {source}")
        return root / source

    @app.get("/api/health")
    def health() -> dict:
        tools = {"catalog": stack.tools.catalog_digest, "warnings": stack.tools.warnings} if stack is not None else None
        return {"ok": True, "agent": getattr(agent, "name", ""), "scenarios": len(scenarios), "tools": tools}

    @app.post("/api/tools/reload")
    def reload_tools() -> dict:
        """도구 서버의 tools/list를 다시 받는다 (docs/08 4-1) — 진행 중인 통화는 자기 스냅샷을 계속 쓴다.
        도구 서버 쪽 카탈로그 리로드(POST /reload)를 먼저 해 둘 것."""
        if stack is None:
            raise HTTPException(400, "도구 스택 없이 뜬 서버")
        from csagent.agent.workers.specs import WORKERS
        return {**stack.tools.reload(), "missing_for_workers": stack.tools.missing_for(WORKERS)}

    @app.get("/api/sessions/{session_id}/crm")
    def session_crm(session_id: str) -> dict:
        """상담 화면의 고객 정보 — 도메인 API의 현재 데이터 (실행 결과가 반영된 상태, docs/08 11-1 6번)."""
        s = _session(store, session_id)
        data = stack.fixtures.get(s.state.ctx.customer_id) if stack is not None else None
        return {"customer_id": s.state.ctx.customer_id, "data": data}

    @app.get("/api/scenarios")
    def list_scenarios() -> list[dict]:
        return [{"id": s.id, "task": s.task, "behavior": s.behavior,
                 "customer_name": s.context.customer.get("name", ""), "customer": s.context.customer,
                 "direction": s.context.direction, "today": s.context.today,
                 "goal": s.customer.get("goal", ""), "opening": s.customer.get("opening", ""),
                 "orders": s.context.customer.get("orders", [])} for s in scenarios]

    @app.get("/api/callers")
    def list_callers() -> list[dict]:
        return [c.view() for c in callers]

    @app.post("/api/sessions")
    def create_session(req: SessionRequest) -> dict:
        sc = by_id.get(req.scenario_id)
        if sc is None:
            raise HTTPException(404, f"시나리오 없음: {req.scenario_id}")
        caller = callers_by_id.get(sc.id)
        if req.situation and not (caller and any(x.id == req.situation for x in caller.situations)):
            raise HTTPException(404, f"상황 없음: {req.situation}")
        ctx = CallContext(direction=sc.context.direction, audience=sc.context.audience,
                          companies=sc.context.companies, service=sc.context.service,
                          customer=dict(sc.context.customer),
                          today=sc.context.today)
        s = store.create(ctx, sc.id, req.mode, req.situation)
        return {"session_id": s.id, "scenario_id": sc.id, "mode": s.mode, "situation": s.situation, "today": ctx.today,
                "customer": ctx.customer, "greeting": greeting_of(s.state.history)}

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        s = _session(store, session_id)
        return {"session_id": s.id, "scenario_id": s.scenario_id, "mode": s.mode, "history": s.state.history,
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
            for name, data in turn_events(s, req.utterance):
                yield _sse(name, data)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.websocket("/api/sessions/{session_id}/voice")
    async def voice(ws: WebSocket, session_id: str, degrade: str = "") -> None:
        """실시간 음성 통화 (csagent/voice/session.py). 연결이 끊기면 통화가 끝난 것으로 저장한다."""
        from csagent.voice.session import VoiceCall, VoiceClient

        await ws.accept()
        s = store.get(session_id)
        if s is None or s.ended or s.mode != "voice" or not (voice_url or make_voice):
            await ws.send_json({"type": "error", "message": "음성 통화를 열 수 없습니다 (세션·모드·음성 서버 확인)"})
            await ws.close()
            return
        client, endpointer = make_voice() if make_voice else (VoiceClient(voice_url), None)
        call = VoiceCall(ws, s, agent, client, turn_events=turn_events, on_ended=save_call,
                         endpointer=endpointer, degrade=degrade, silence_s=silence_s)
        event(log, "voice_open", call_id=s.state.call_id, scenario=s.scenario_id, degrade=degrade)
        if (text := greeting_of(s.state.history)) and not s.replies and not s.voice_turns:
            await call.greet(text)
        call.start_silence_watch()
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await call.on_audio(msg["bytes"])
                elif msg.get("text"):
                    ctl = json.loads(msg["text"])
                    if ctl.get("type") == "playing":
                        call.set_playing(bool(ctl.get("on")))
                    elif ctl.get("type") in ("played", "cut"):
                        call.on_playback(ctl)
                    elif ctl.get("type") == "hangup":
                        break
        except WebSocketDisconnect:
            pass
        finally:
            await call.close()
            await client.aclose()
            if s.replies or s.voice_turns:
                save_call(s, "agent_ended" if s.ended else "customer_hangup")

    @app.get("/api/llm_calls/{llm_call_id}")
    def llm_call(llm_call_id: str, source: Optional[str] = None) -> dict:
        rec = find_llm_call(source_dir(source), llm_call_id)
        if rec is None:
            raise HTTPException(404, f"LLM 호출 기록 없음: {llm_call_id}")
        return rec

    @app.get("/api/recordings")
    def recordings(source: Optional[str] = None) -> list[dict]:
        return list_recordings(source_dir(source))

    @app.get("/api/recordings/{rid}")
    def recording(rid: str, source: Optional[str] = None) -> dict:
        rec = load_recording(source_dir(source), rid)
        if rec is None:
            raise HTTPException(404, f"기록 없음: {rid}")
        return rec

    @app.post("/api/tts")
    async def tts(req: TTSRequest) -> Response:
        from csagent.voice.session import VoiceClient
        if not voice_url:
            raise HTTPException(503, "음성 서버 주소가 없습니다")
        client = VoiceClient(voice_url, voice=req.voice)
        try:
            wav, gen_ms = await client.tts(req.text)
        finally:
            await client.aclose()
        return Response(wav, media_type="audio/wav", headers={"X-Gen-Ms": str(gen_ms)})

    def demos_path() -> Path:
        return root / DEMOS_FILE

    @app.get("/api/demos")
    def demos() -> list[dict]:
        path = demos_path()
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    @app.post("/api/demos")
    def add_demo(req: DemoRequest) -> list[dict]:
        """녹화 재생 탭에서 고른 기록을 데모 목록에 올린다 — 기록이 실제로 있어야 한다 (응답을 지어내지 않음)."""
        if load_recording(source_dir(req.source), req.record_id) is None:
            raise HTTPException(404, f"기록 없음: {req.source} {req.record_id}")
        if not req.title.strip():
            raise HTTPException(422, "제목이 비어 있습니다")
        return save_demos(req.source, req.record_id, req.model_dump())

    @app.delete("/api/demos")
    def remove_demo(source: str, record_id: str) -> list[dict]:
        return save_demos(source, record_id, None)

    def save_demos(source: str, record_id: str, item: Optional[dict]) -> list[dict]:
        """같은 기록의 항목을 빼고 item이 있으면 뒤에 붙인다 (제목 수정 = 다시 올리기)."""
        with demos_lock:
            items = [d for d in demos() if not (d["source"] == source and d["record_id"] == record_id)]
            if item:
                items.append(item)
            demos_path().parent.mkdir(parents=True, exist_ok=True)
            demos_path().write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return items

    @app.get("/api/monitor/sources")
    def monitor_sources() -> dict:
        own = f"{log_dir.parent.name}/{log_dir.name}" if log_dir else None
        return {"server": own, "sources": list_sources(root)}

    @app.get("/api/monitor")
    def monitor(source: Optional[str] = None) -> dict:
        return aggregate(source_dir(source))

    if web_dir and web_dir.exists():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


def create_default_app() -> FastAPI:
    """실제 구성요소로 조립한 앱 (uvicorn 진입점)."""
    from datetime import datetime

    from csagent.config import load_settings
    from csagent.factory import build_agent
    from csagent.llm_log import LLMCallLog
    from csagent.obs import LLM_CALLS_FILE, setup_logging

    log_dir = setup_logging(ROOT / "logs" / f"{datetime.now():%Y%m%d}")   # 서버 시작 날짜별 로그 폴더
    llm_log = LLMCallLog(log_dir / LLM_CALLS_FILE)
    from csagent.server.callers import load_callers

    from csagent.stack import record_versions, remote_stack

    stack = remote_stack()   # 도구 서버·도메인 API가 떠 있어야 한다 (scripts/run_tools.py) — 없으면 여기서 실패
    # 버전 기록 — 하루 폴더에 서버를 여러 번 띄우므로 기동마다 한 줄. 통화 기록에도 같은 버전을 붙인다
    meta = {"started_at": time.time(), **record_versions(stack, log_dir)}
    with (log_dir / "server_meta.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    # scenarios_live/: 실제 테스트 탭 발신자 (평가에는 안 들어감)
    return create_app(build_agent(log=llm_log, tools=stack.tools), load_scenarios(ROOT / "scenarios"),
                      callers=load_callers(ROOT / "scenarios_live"), log_dir=log_dir,
                      voice_url=load_settings().voice_url, stack=stack,
                      version={k: meta.get(k) for k in ("commit", "dirty", "diff_sha", "catalog")})


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
        "pending": {"kind": p.kind, "items": [{"action": a, "target": t} for a, t in p.items],
                    "turn": p.turn} if p else None,
        "confirmations": [{"action": c.action, "target": c.target, "turn": c.turn} for c in state.confirmations],
        "executed": sorted(state.executed),
        "active": {"task": state.active.name, "slots": state.active.slots} if state.active else None,
        "queue": [t.name for t in state.queue],
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
