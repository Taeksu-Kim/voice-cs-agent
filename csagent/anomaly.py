"""이상 탐지 — 끝난 통화 기록에서 분석할 후보를 고르고 케이스 묶음을 만든다 (docs/10 4절).

탐지는 후보를 고르는 일이라 정밀할 필요는 없다 (거르는 건 분석하는 쪽). 다만 신호마다 「이상하다고 볼 이유」가 분명해야 한다.

- 확실한 신호 (trace 규칙): 턴 오류 · 결과 모름 · 대체 문장(못 들었어요·문제가 생겼어요)은 bad,
  도구 실패(백엔드 탓일 수 있다 — 대응이 맞았는지는 따로) · 반복 · 150자 초과 · 턴 한도 종료 · 미해결 종료는 warn
- 판정기 신호: 평가 기록은 이미 채점된 실패 항목, 서버 통화는 judge를 넘기면 근거 판정(v4)을 돌린다
- 가드레일 발동(막거나 고친 곳)은 신호가 아니라 맥락이다 — 결함은 이미 막혔다. 통화별 발동 수만 세어 둔다 (지시문 개선 후보)

케이스 묶음 <out>/<키>/: case.json(출처·신호·가드레일·버전) · trace.md(mermaid) · llm_calls.jsonl(그 통화의 LLM 원문) ·
repro.yaml(재현 명세 초안 — 구간·술어는 분석하는 쪽이 고친다)
"""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from csagent.agent.loop import FALLBACK, TOOL_STUCK
from csagent.monitoring import MAX_REPLY_CHARS
from csagent.obs import CALLS_FILE, LLM_CALLS_FILE
from csagent.tracegraph import coords, severity, to_mermaid

REPEAT_SIMILARITY = 0.9          # 평가 no_repeat과 같은 기준
# 이미 규칙 신호가 위치까지 잡는 채점 항목 · 시뮬레이터 쪽 문제라 에이전트 결함이 아닌 진단
_COVERED = ("no_repeat", "voice_length")


@dataclass
class Signal:
    kind: str           # turn_error · execution_unknown · tool_failed · fallback · repeat · over_len · max_turns · unresolved · check:<항목>
    level: str          # bad(결함이 났다) | warn(확인 필요)
    at: str             # 좌표 (t3 · t3.s2) — 모르면 ""
    detail: str


def _turn_at(at: str) -> str:
    """t3.s2.gate → t3.s2"""
    return ".".join(at.split(".")[:2])


def _agent_turns(history: list) -> list[tuple[int, str, str]]:
    """(턴, 직전 고객 말, 상담원 말). 고객 말마다 턴이 하나 늘고, 첫 고객 말 전(인사)은 뺀다."""
    out, turn, asked = [], 0, ""
    for role, text in history:
        if role == "customer":
            turn, asked = turn + 1, text
        elif turn:
            out.append((turn, asked, text))
    return out


def _similar(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a, b).ratio() >= REPEAT_SIMILARITY


def rule_signals(row: dict[str, Any]) -> list[Signal]:
    events = coords(row.get("trace") or [])
    out = []
    for e in events:
        k, d, at = e["kind"], e["data"], _turn_at(e["at"])
        if k == "turn_error":
            out.append(Signal(k, "bad", at, d.get("error", "")))
        elif k == "execution_unknown":
            out.append(Signal(k, "bad", at, f"{d.get('name')} {d.get('args')}"))
        elif k == "tool_result" and d.get("ok") is False:
            out.append(Signal("tool_failed", "warn", at, f"{d.get('name')}: {d.get('error') or d.get('reason') or ''}"))
        elif k == "spoken" and any(x in d.get("text", "") for x in (FALLBACK, TOOL_STUCK)):
            out.append(Signal("fallback", "bad", at, d["text"][:60]))
    turns = _agent_turns(row.get("turns") or row.get("history") or [])
    for (_, qa, a), (turn, qb, b) in zip(turns, turns[1:]):
        if _similar(a, b) and not _similar(qa, qb):     # 같은 질문에 같은 답은 정상
            out.append(Signal("repeat", "warn", f"t{turn}", b[:80]))
    for turn, _, text in turns:
        if len(text) > MAX_REPLY_CHARS:
            out.append(Signal("over_len", "warn", f"t{turn}", f"{len(text)}자"))
    if row.get("end_reason") == "max_turns":
        out.append(Signal("max_turns", "warn", "", "턴 한도로 끝남"))
    if row.get("disposition") == "unresolved" and "checks" not in row:   # 평가 기록은 disposition 채점이 기대값과 비교한다
        out.append(Signal("unresolved", "warn", "", "미해결로 끝남"))
    return out


def check_signals(row: dict[str, Any]) -> list[Signal]:
    """평가 기록의 실패한 채점 항목. 근거 판정은 턴마다 하나씩 (상세에 「3턴: 주장」)."""
    out = []
    for c in row.get("checks") or []:
        if c.get("passed") is not False or c["name"] in _COVERED:
            continue
        if c["name"] == "grounded":
            for part in c.get("detail", "").split("; "):
                m = re.match(r"(\d+)턴: (.*)", part)
                out.append(Signal("check:grounded", "bad", f"t{m.group(1)}" if m else "", m.group(2) if m else part))
        else:
            out.append(Signal(f"check:{c['name']}", "bad", "", c.get("detail", "")))
    return out


def judge_signals(row: dict[str, Any], judge: Any) -> list[Signal]:
    """서버 통화(채점 없음)에 근거 판정 v4를 돌린다. judge = LLMJudge."""
    from csagent.agent.contracts import AgentReply, CallContext, ToolCall
    from csagent.agent.state import CallState
    from csagent.agent.trace import Trace
    from csagent.evaluation.runner import Transcript

    state = CallState(ctx=CallContext())
    state.history, state.trace = [tuple(x) for x in row.get("history") or []], Trace.load(row.get("trace") or [])
    replies = [AgentReply(text=t) for _, _, t in _agent_turns(state.history)]
    tr = Transcript(row.get("scenario") or "", state, replies=replies)
    return [Signal("check:grounded", "bad", f"t{g['turn']}", g["claim"])
            for g in judge.grounded_turns(tr) if g["unsupported"]]


def guard_counts(row: dict[str, Any]) -> dict[str, int]:
    """가드레일이 막거나 고친 곳 (tracegraph의 guard). 게이트 차단은 사유별로."""
    c = Counter()
    for e in row.get("trace") or []:
        if severity(e) == "guard":
            c[f"gate:{e['data'].get('reason')}" if e["kind"] == "gate" else e["kind"]] += 1
    return dict(c)


# -- 기록 읽기 --

def load_rows(source: Path) -> list[tuple[str, dict[str, Any]]]:
    """(키, 기록). 평가 폴더 키 = 시나리오:반복, 서버 폴더 키 = call_id — 재현기 load_case와 같은 키."""
    if (source / "results.jsonl").exists():
        rows = [json.loads(l) for l in (source / "results.jsonl").open(encoding="utf-8") if l.strip()]
        return [(f"{r['scenario_id']}:{r['repeat']}", r) for r in rows]
    if (source / CALLS_FILE).exists():
        rows = [json.loads(l) for l in (source / CALLS_FILE).open(encoding="utf-8") if l.strip()]
        return [(r["call_id"], r) for r in rows]
    return []


def detect(source: Path, judge: Any = None) -> list[dict[str, Any]]:
    """기록 폴더 하나 → 통화별 {key, signals, guards}. 서버 통화는 judge가 있으면 근거 판정도."""
    out = []
    for key, row in load_rows(source):
        signals = rule_signals(row) + (check_signals(row) if "checks" in row
                                       else judge_signals(row, judge) if judge else [])
        out.append({"source": source, "key": key, "row": row, "signals": signals, "guards": guard_counts(row)})
    return out


# -- 케이스 묶음 --

_STEP_PREDS = ("len_over", "regex", "tool", "no_tool")


def _spec(case: str, sig: Signal, row: dict[str, Any]) -> dict[str, Any]:
    """신호 → 재현 명세 초안. 술어는 신호를 그대로 옮긴 것 — 분석 후 좁히거나 바꾼다.
    턴 시작 상태가 없는 기록(10/21 전)은 step 모드만 된다 — 그 턴 첫 스텝의 LLM 입력을 다시 보낸다."""
    turn = sig.at.split(".")[0] or "t1"
    fail = {
        "over_len": ("turn", [{"len_over": MAX_REPLY_CHARS}]),
        "repeat": ("turn", [{"repeat": True}]),
        "fallback": ("turn", [{"regex": "잘 못 들었어요|문제가 생겼어요"}]),
        "turn_error": ("turn", [{"event": "turn_error"}]),
        "execution_unknown": ("turn", [{"event": "execution_unknown"}]),
        "max_turns": ("continue", [{"check": "finished"}]),
        "unresolved": ("continue", [{"disposition": "unresolved"}]),
    }.get(sig.kind)
    if fail is None and sig.kind.startswith("check:"):
        fail = ("continue", [{"check": sig.kind[6:]}])
    mode, fail_if = fail or ("turn", [])
    at, note = turn if sig.at else "t1", f"탐지 신호 {sig.kind} @{sig.at or '-'}: {sig.detail}"
    if not any(e["kind"] == "state" for e in row.get("trace") or []):
        mode, at = "step", sig.at if ".s" in sig.at else f"{at}.s1"
        fail_if = [p for p in fail_if if next(iter(p)) in _STEP_PREDS]
        note += " (턴 시작 상태 없는 기록 — step 모드만)"
    return {"case": case, "at": at, "mode": mode, "repeat": 5, "fail_if": fail_if, "note": note}


def _llm_calls(source: Path, row: dict[str, Any]) -> list[str]:
    ids = {e["data"].get("call_id") for e in row.get("trace") or [] if e["kind"] == "llm"}
    path = source / LLM_CALLS_FILE
    if not ids or not path.exists():
        return []
    return [l for l in path.open(encoding="utf-8") if any(f'"{i}"' in l for i in ids)
            and json.loads(l).get("call_id") in ids]


def write_case(out: Path, root: Path, item: dict[str, Any]) -> Path:
    source, key, row, signals = item["source"], item["key"], item["row"], item["signals"]
    case = f"{source.relative_to(root) if source.is_relative_to(root) else source}/{key}"
    d = out / f"{source.name}_{key.replace(':', '_r')}"
    d.mkdir(parents=True, exist_ok=True)
    version = row.get("version")
    if version is None and (source / "run_meta.json").exists():
        meta = json.loads((source / "run_meta.json").read_text(encoding="utf-8"))
        version = {k: meta.get(k) for k in ("commit", "dirty", "diff_sha", "models") if k in meta}
    (d / "case.json").write_text(json.dumps({
        "case": case, "scenario": row.get("scenario_id") or row.get("scenario"),
        "disposition": row.get("disposition"), "end_reason": row.get("end_reason"),
        "signals": [asdict(s) for s in signals], "guards": item["guards"], "version": version,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    failed = [f"{s.kind} {s.at}".strip() for s in signals]
    (d / "trace.md").write_text(f"# {case}\n\n```mermaid\n{to_mermaid(row.get('trace') or [], case, failed)}\n```\n",
                                encoding="utf-8")
    (d / "llm_calls.jsonl").write_text("".join(_llm_calls(source, row)), encoding="utf-8")
    first = sorted(signals, key=lambda s: (s.level != "bad", s.at or "t99"))[0]
    (d / "repro.yaml").write_text(yaml.safe_dump(_spec(case, first, row), allow_unicode=True, sort_keys=False),
                                  encoding="utf-8")
    return d
