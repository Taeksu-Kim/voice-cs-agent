"""데모 탭의 재생 원본 — 실제로 돌았던 통화 기록을 턴 단위로 정리한다. 응답을 새로 만들지 않는다.

원본 두 가지:
- 서버 로그 logs/<날짜>/calls.jsonl  — 웹에서 한 통화. mode=voice면 voice_turns에 STT 결과·문장·실측 지연이 있다
- 평가 실행 runs/<실행>/results.jsonl — 고객은 LLM 시뮬레이터, 텍스트로만 돈 통화 (음성 지연 기록 없음)

화면은 kind로 둘을 구분해 표시한다 (voice: 실측 음성 지연 / text: 텍스트 기록, 재생 간격은 임의).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

from csagent.obs import CALLS_FILE

RESULTS_FILE = "results.jsonl"
_CHUNKS: dict[str, dict] = {}


def _sources(chunk_ids: list[str]) -> list[dict]:
    """기록엔 chunk_id만 있다 — 지식 베이스에서 제목·URL을 찾아 붙인다 (데모 채팅창의 「출처」)."""
    if not _CHUNKS:
        from csagent.config import load_settings
        path = load_settings().processed_dir / "chunks.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                c = json.loads(line)
                _CHUNKS[c["chunk_id"]] = {"title": c.get("title", ""), "url": c.get("url", "")}
    return [{"chunk_id": cid, **_CHUNKS[cid]} for cid in chunk_ids if cid in _CHUNKS]


def _records(log_dir: Path) -> Iterator[tuple[str, dict]]:
    for name in (CALLS_FILE, RESULTS_FILE):
        path = log_dir / name
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                rec = json.loads(line)
                rid = rec.get("call_id") or f"{rec.get('scenario_id')}#{rec.get('repeat', 0)}#{i}"
                yield rid, rec


def list_recordings(log_dir: Path) -> list[dict[str, Any]]:
    out = []
    for rid, rec in _records(log_dir):
        voice = bool(rec.get("voice_turns"))
        out.append({"id": rid, "scenario": rec.get("scenario") or rec.get("scenario_id"),
                    "situation": rec.get("situation", ""),
                    "kind": "voice" if voice else "text", "mode": rec.get("mode", "eval"),
                    "created_at": rec.get("created_at", ""), "disposition": rec.get("disposition"),
                    "passed": rec.get("passed"), "end_reason": rec.get("end_reason"),
                    "turns": len(rec["voice_turns"]) if voice else
                    sum(1 for r, _ in rec.get("history") or rec.get("turns") or [] if r == "customer")})
    return out


def load_recording(log_dir: Path, rid: str) -> Optional[dict[str, Any]]:
    for got, rec in _records(log_dir):
        if got == rid:
            return _normalize(rid, rec)
    return None


def greeting_of(history) -> str:
    """기록 맨 앞의 에이전트 발화 = 인바운드 고정 인사 (interface.new_call). 없으면 ""."""
    return history[0][1] if history and history[0][0] == "agent" else ""


def _normalize(rid: str, rec: dict) -> dict[str, Any]:
    trace = rec.get("trace") or []
    by_turn: dict[int, list[dict]] = {}
    for e in trace:
        by_turn.setdefault(e.get("turn", 0), []).append(e)
    replies = rec.get("replies") or []
    history = rec.get("history") or rec.get("turns") or []
    greeting = greeting_of(history)
    turns: list[dict] = []
    if rec.get("voice_turns"):
        done = iter(replies)
        for vt in rec["voice_turns"]:
            reply = next(done) if vt.get("completed") else None
            turns.append({"no": vt.get("turn"), "customer": vt.get("stt", {}).get("text", ""),
                          "utterance": vt.get("utterance", ""),
                          "agent": reply["text"] if reply else " ".join(s["text"] for s in vt["sentences"]),
                          "sentences": vt.get("sentences", []), "latency": vt.get("latency", {}),
                          "completed": vt.get("completed", False), "barge_in": vt.get("barge_in", False),
                          "reply": reply, "trace": _trace(by_turn.get(vt.get("turn"), []))})
        kind = "voice"
    else:
        customers = [t for r, t in history if r == "customer"]
        agents = [t for r, t in history if r == "agent"][1 if greeting else 0:]
        for i, text in enumerate(customers):
            reply = replies[i] if i < len(replies) else None
            agent = reply["text"] if reply else (agents[i] if i < len(agents) else "")
            turns.append({"no": i + 1, "customer": text, "utterance": text, "agent": agent,
                          "sentences": [], "latency": {}, "completed": reply is not None, "barge_in": False,
                          "reply": reply, "trace": _trace(by_turn.get(i + 1, []))})
        kind = "text"
    for t in turns:
        t["sources"] = _sources((t["reply"] or {}).get("retrieved", []))
    return {"id": rid, "scenario": rec.get("scenario") or rec.get("scenario_id"), "kind": kind,
            "situation": rec.get("situation", ""),
            "mode": rec.get("mode", "eval"), "created_at": rec.get("created_at", ""),
            "disposition": rec.get("disposition"), "end_reason": rec.get("end_reason"),
            "passed": rec.get("passed"), "greeting": greeting, "turns": turns}


def _trace(events: list[dict]) -> list[dict]:
    """화면의 추적 기록 형식 (app.py _trace_view와 같게): 턴 시작부터의 ms."""
    if not events:
        return []
    t0 = events[0].get("t", 0)
    return [{"kind": e["kind"], "ms": round((e.get("t", t0) - t0) * 1000),
             "data": {k: v for k, v in e.get("data", {}).items() if k != "texts"}} for e in events]
