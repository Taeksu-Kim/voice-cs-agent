"""모니터링 집계 — 로그 폴더(events.jsonl·llm_calls.jsonl)를 읽어 지표를 만든다.

CLI 모니터(scripts/monitor.py)와 웹 모니터(/api/monitor)가 같은 집계를 쓴다.
파일은 실행 중에도 계속 자라므로, 새로 붙은 부분만 읽는다 (파일별 읽은 위치를 기억).
llm_calls.jsonl은 크기가 커서 지표에 필요한 필드만 남기고, 원문은 호출 id → 파일 위치로 필요할 때 꺼낸다.

지표
- 평가: 시나리오별 통과/실행 수, 실패 항목, 시뮬레이터 이탈 (scenario_result 이벤트)
- 통화: 시작·종료 수, 통화 결과(disposition) 분포
- 턴: 응답 지연(전체·첫 글자) p50/p90/max, 150자 초과 응답 수, 에이전트 토큰
- LLM: 출처별(agent·router·customer·judge) 호출 수·입출력 토큰·지연·오류, 최근 60초 토큰
- 사건: 폴백(사유별), 절차 강제, 스텝 소진, 라우팅 실패, 되묻기, 업무 전환, 도구 결말(ok·confirm·blocked·failed)과 차단 사유
- 최근 경고: WARNING 이상 이벤트 최근 30건
"""

from __future__ import annotations

import json
import statistics as st
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from csagent.obs import EVENTS_FILE, LLM_CALLS_FILE

MAX_REPLY_CHARS = 150          # 평가의 voice_length 기준과 같다
_LLM_FIELDS = ("call_id", "ts", "source", "scenario", "repeat", "prompt_tokens", "completion_tokens", "ms", "error")


@dataclass
class _Tail:
    """JSONL 파일을 이어 읽는다. records는 지금까지 읽은 줄(가공 후), offsets는 id → 줄 시작 위치."""
    pos: int = 0
    records: list[dict] = field(default_factory=list)
    offsets: dict[str, int] = field(default_factory=dict)


_tails: dict[Path, _Tail] = {}
_lock = threading.Lock()


def _read_new(path: Path, keep: Optional[tuple[str, ...]] = None, id_key: Optional[str] = None) -> list[dict]:
    with _lock:
        tail = _tails.setdefault(path, _Tail())
        if not path.exists():
            return tail.records
        if path.stat().st_size < tail.pos:      # 파일이 새로 만들어졌으면 처음부터
            tail = _tails[path] = _Tail()
        with path.open("rb") as f:
            f.seek(tail.pos)
            while True:
                start = f.tell()
                line = f.readline()
                if not line or not line.endswith(b"\n"):   # 쓰는 중인 마지막 줄은 다음에 읽는다
                    f.seek(start)
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if id_key and rec.get(id_key):
                    tail.offsets[rec[id_key]] = start
                tail.records.append({k: rec.get(k) for k in keep} if keep else rec)
            tail.pos = f.tell()
        return tail.records


def find_llm_call(log_dir: Path, llm_call_id: str) -> Optional[dict]:
    """LLM 호출 하나의 원문 (입력 메시지·출력)."""
    path = Path(log_dir) / LLM_CALLS_FILE
    _read_new(path, _LLM_FIELDS, id_key="call_id")
    offset = _tails[path].offsets.get(llm_call_id)
    if offset is None:
        return None
    with path.open("rb") as f:
        f.seek(offset)
        return json.loads(f.readline())


def aggregate(log_dir: Path, now: Optional[float] = None) -> dict[str, Any]:
    log_dir = Path(log_dir)
    events = _read_new(log_dir / EVENTS_FILE)
    llm = _read_new(log_dir / LLM_CALLS_FILE, _LLM_FIELDS, id_key="call_id")
    now = now or time.time()
    by_event: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_event[e.get("event")].append(e)

    turns = by_event["turn"]
    tools = by_event["tool"]
    out: dict[str, Any] = {
        "source": str(log_dir), "updated": now,
        "first_ts": min((r["ts"] for r in events + llm if r.get("ts")), default=None),
        "last_ts": max((r["ts"] for r in events + llm if r.get("ts")), default=None),
        "calls": {
            "started": len(by_event["call_start"]), "ended": len(by_event["call_end"]),
            "dispositions": dict(Counter(e.get("disposition") for e in by_event["call_end"])),
        },
        "turns": {
            "n": len(turns),
            "ms_total": _pct([e.get("ms_total", 0) for e in turns]),
            "ms_first_token": _pct([e.get("ms_first_token", 0) for e in turns if e.get("ms_first_token")]),
            "llm_steps": _pct([e.get("llm_steps", 0) for e in turns]),
            "over_chars": sum(1 for e in turns if e.get("chars", 0) > MAX_REPLY_CHARS),
            "prompt_tokens": sum(e.get("prompt_tokens", 0) for e in turns),
            "completion_tokens": sum(e.get("completion_tokens", 0) for e in turns),
        },
        "llm": _llm_stats(llm, now),
        "incidents": {
            "fallback": dict(Counter(e.get("reason") for e in by_event["fallback"])),
            "step_enforced": len(by_event["step_enforced"]),
            "max_steps": len(by_event["max_steps"]),
            "route_failed": len(by_event["route_failed"]),
            "ask_again": len(by_event["ask_again"]),
            "task_switch": dict(Counter(e.get("by") for e in by_event["task_switch"])),
            "llm_error": len(by_event["llm_error"]),
            "turn_error": len(by_event["turn_error"]),
        },
        "tools": {
            "status": dict(Counter(e.get("status") for e in tools)),
            "blocked": dict(Counter(f"{e.get('tool')}:{e.get('reason')}" for e in tools
                                    if e.get("status") == "blocked")),
            "failed": dict(Counter(f"{e.get('tool')}:{e.get('reason')}" for e in tools
                                   if e.get("status") == "failed")),
            "by_tool": dict(Counter(e.get("tool") for e in tools)),
        },
        "warnings": [e for e in events if e.get("level") in ("WARNING", "ERROR", "CRITICAL")][-30:],
    }
    results = by_event["scenario_result"]
    if results:
        per: dict[str, dict] = {}
        for e in results:
            p = per.setdefault(e.get("scenario"), {"passed": 0, "n": 0, "failed": Counter(), "sim_drift": 0})
            p["n"] += 1
            p["passed"] += bool(e.get("passed"))
            p["failed"].update(e.get("failed") or [])
            p["sim_drift"] += bool(e.get("sim_drift"))
        out["eval"] = {
            "done": len(results), "passed": sum(bool(e.get("passed")) for e in results),
            "sim_drift": sum(bool(e.get("sim_drift")) for e in results),
            "scenarios": {k: {**v, "failed": dict(v["failed"])} for k, v in per.items()},
        }
    return out


def _llm_stats(llm: list[dict], now: float) -> dict[str, Any]:
    by_src: dict[str, list[dict]] = defaultdict(list)
    for c in llm:
        by_src[c.get("source") or "?"].append(c)
    recent = [c for c in llm if c.get("ts") and now - c["ts"] <= 60]
    return {
        "by_source": {src: {
            "calls": len(cs),
            "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in cs),
            "completion_tokens": sum(c.get("completion_tokens") or 0 for c in cs),
            "ms": _pct([c["ms"] for c in cs if c.get("ms") is not None]),
            "errors": sum(1 for c in cs if c.get("error") and c["error"] != "aborted"),
        } for src, cs in sorted(by_src.items())},
        "last_60s": {"calls": len(recent),
                     "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in recent),
                     "completion_tokens": sum(c.get("completion_tokens") or 0 for c in recent)},
    }


def _pct(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0, "p90": 0, "max": 0}
    v = sorted(values)
    return {"p50": round(st.median(v)), "p90": round(v[int(len(v) * .9)]), "max": round(v[-1])}


def list_sources(root: Path, limit: int = 30) -> list[dict[str, Any]]:
    """모니터링할 수 있는 로그 폴더: 평가 실행(runs/*)과 서버 로그(logs/*), 최근 것부터."""
    dirs = [d for base in ("runs", "logs") if (root / base).exists()
            for d in (root / base).iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: max((f.stat().st_mtime for f in d.iterdir()), default=d.stat().st_mtime),
              reverse=True)
    return [{"id": f"{d.parent.name}/{d.name}", "kind": "eval" if d.parent.name == "runs" else "server",
             "has_events": (d / EVENTS_FILE).exists()} for d in dirs[:limit]]
