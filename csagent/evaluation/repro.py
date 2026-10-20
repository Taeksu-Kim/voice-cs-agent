"""재현기 — 기록된 통화의 한 지점(좌표)을 N번 다시 돌려 같은 실패가 몇 번 나오는지 센다 (docs/10 5절).

구간 선택은 사람·Claude가 하고(재현 명세), 실행은 여기서 고정된 방식으로 한다 — 수정 전후를 같은 조건으로 비교하려고.

    case: runs/20261021_010852_C/s3_b0_defect:0      # 평가 기록 (시나리오:반복) 또는 logs/<날짜>/<call_id>
    at: t1                                            # 좌표 — turn·continue는 턴 단위, step은 t1.s2처럼 스텝까지
    mode: turn
    repeat: 5
    fail_if: [{len_over: 150}, {repeat: true}]        # 하나라도 맞으면 실패. 수정 전에 정하고 수정 후에도 그대로

모드
- step:     그 스텝의 LLM 입력 원문(llm_calls.jsonl)을 그대로 다시 보낸다 — 지금 코드·지시문과 무관, 모델 흔들림 확인용
- turn:     턴 시작 상태(trace의 state 스냅샷)·이력·도메인 데이터를 되살려 그 턴의 고객 발화를 지금 코드로 다시 처리
- continue: turn 뒤로 통화를 끝까지 이어 간다 (채점 항목 check: 를 쓸 수 있다).
            customer: recorded(기본) = 기록된 고객 말을 순서대로 (에이전트 변화만 본다 — 말이 바뀌면 대화가 어긋날 수 있다)
                      simulator = 고객 시뮬레이터 (대화 흐름 전체의 강건성)

fail_if 술어 (turn·continue는 다시 만든 턴들, step은 그 스텝 출력에 적용)
  len_over: N      말이 N자 초과 (채점 voice_length와 같은 기준: 한 턴에 들은 말 전체)
  repeat: true     바로 앞 상담원 말과 거의 같은 말 (고객도 같은 말을 했으면 제외 — 채점 no_repeat과 같은 규칙)
  regex: 패턴      말에 패턴이 있음
  tool: 이름 / no_tool: 이름    그 도구를 불렀음 / 안 불렀음
  event: 종류      그 trace 이벤트가 생김 (unheard_drop, promise_nudge, turn_error …)
  gate: 사유       게이트가 그 사유로 막음
  disposition: 값  통화 결과가 그 값
  check: 이름      채점 항목 실패 (통화 전체 기준 — continue에서 쓴다. 판정기 항목은 judge가 있어야 한다)
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from csagent.agent.contracts import AgentReply, ToolCall, TurnInput
from csagent.agent.state import CallState
from csagent.agent.trace import Trace
from csagent.config import ROOT
from csagent.evaluation.runner import Transcript, _converse
from csagent.evaluation.scenario import Scenario, load_scenarios
from csagent.evaluation.scoring import REPEAT_SIMILARITY, score
from csagent.evaluation.simulator import ScriptedCustomer
from csagent.tracegraph import coords

SCENARIO_DIRS = ("scenarios", "scenarios_heldout")


class ReproError(ValueError):
    pass


@dataclass
class ReproSpec:
    case: str
    at: str
    mode: str = "turn"
    repeat: int = 5
    fail_if: list[dict[str, Any]] = field(default_factory=list)
    customer: str = "recorded"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReproSpec":
        spec = cls(d["case"], str(d["at"]), d.get("mode", "turn"), int(d.get("repeat", 5)), list(d.get("fail_if") or []),
                   d.get("customer", "recorded"))
        if spec.customer not in ("recorded", "simulator"):
            raise ReproError(f"customer는 recorded | simulator: {spec.customer}")
        if spec.mode not in ("step", "turn", "continue"):
            raise ReproError(f"mode는 step | turn | continue: {spec.mode}")
        if not spec.fail_if:
            raise ReproError("fail_if가 없다 — 무엇을 실패로 볼지 수정 전에 정한다")
        return spec

    def coordinate(self) -> tuple[int, Optional[int]]:
        m = re.fullmatch(r"t(\d+)(?:\.s(\d+))?(?:\..*)?", self.at)
        if not m:
            raise ReproError(f"좌표 형식: t<턴>[.s<스텝>] — {self.at}")
        return int(m[1]), (int(m[2]) if m[2] else None)


# -- 기록 찾기 --

def load_case(case: str) -> tuple[Path, dict[str, Any]]:
    """case = <폴더>/<키>. 평가 폴더(results.jsonl)면 키는 시나리오[:반복], 서버 폴더(calls.jsonl)면 call_id."""
    source, key = Path(case).parent, Path(case).name
    source = source if source.is_absolute() else ROOT / source
    if (source / "results.jsonl").exists():
        sid, _, rep = key.partition(":")
        for line in (source / "results.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            if row["scenario_id"] == sid and (not rep or row["repeat"] == int(rep)):
                return source, row
    if (source / "calls.jsonl").exists():
        for line in (source / "calls.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            if row["call_id"] == key:
                return source, row
    raise ReproError(f"기록 없음: {case}")


def find_scenario(row: dict[str, Any]) -> Scenario:
    sid = row.get("scenario_id") or row.get("scenario")
    for d in SCENARIO_DIRS:
        for sc in load_scenarios(ROOT / d):
            if sc.id == sid:
                return sc
    from csagent.server.callers import load_callers   # 실제 테스트 탭 발신자 (서버 통화)
    for c in load_callers(ROOT / "scenarios_live"):
        if c.id == sid:
            return c.to_scenario()
    raise ReproError(f"시나리오·발신자 없음: {sid}")


def _history(row: dict[str, Any]) -> list[tuple[str, str]]:
    return [tuple(x) for x in (row.get("turns") or row.get("history") or [])]


def _replies(row: dict[str, Any]) -> list[AgentReply]:
    out = []
    for r in row.get("replies") or []:
        if "tool_calls" in r:   # 서버 기록 (asdict(AgentReply))
            calls = [ToolCall(c["name"], c["args"], c.get("result")) for c in r["tool_calls"]]
        else:                   # 평가 기록 (tools: [이름, 인자, 결과])
            calls = [ToolCall(n, a, res) for n, a, res in r.get("tools") or []]
        out.append(AgentReply(text=r["text"], tool_calls=calls, retrieved=r.get("retrieved") or [],
                              disposition=r.get("disposition")))
    return out


# -- 상태 되살리기 --

def restore(stack: Any, sc: Scenario, row: dict[str, Any], turn: int) -> tuple[CallState, str, str]:
    """turn 시작 직전으로 되돌린 통화 상태 + 그 턴의 고객 발화·이벤트."""
    events = coords(row["trace"])
    start = next((e for e in events if e["turn"] == turn and e["kind"] == "turn_start"), None)
    snap = next((e for e in events if e["turn"] == turn and e["kind"] == "state"), None)
    if start is None:
        raise ReproError(f"{turn}턴이 기록에 없다")
    if snap is None:
        raise ReproError(f"{turn}턴 상태 스냅샷 없음 (10/21 전 기록) — step 모드만 된다")
    state = stack.open(sc.context)
    prefix = [e for e in events if e["turn"] < turn]
    state.trace = Trace.load(prefix)
    hist, n = _history(row), 0
    for i, (role, _) in enumerate(hist):
        if role == "customer":
            n += 1
            if n == turn:
                hist = hist[:i]
                break
    state.history, state.turn = list(hist), turn - 1
    state.restore(snap["data"])
    # 앞 턴들이 도메인 데이터를 바꾼 실행(분실신고·링크 발송 등)을 다시 적용 — 그래야 조회 결과가 그때와 같다
    tools, last_args = stack.tools.for_call(state), {}
    for e in prefix:
        if e["kind"] == "tool_request":
            last_args[e["data"]["name"]] = e["data"]["args"]
        elif e["kind"] == "tool_result" and e["data"].get("ok") and e["data"].get("risk") in ("execute", "notify"):
            name = e["data"]["name"]
            out = tools.run(state, name, last_args.get(name, {}))
            if not out.ok:
                raise ReproError(f"앞 턴 실행 다시 적용 실패: {name} {out.error}")
    return state, start["data"]["utterance"], start["data"].get("event", "speech")


# -- 실패 판정 --

def _similar(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a, b).ratio() >= REPEAT_SIMILARITY


def judge_preds(fail_if: list[dict[str, Any]], *, new: list[AgentReply], history: list[tuple[str, str]],
                events: list[dict[str, Any]], disposition: Optional[str] = None,
                failed_checks: Optional[set[str]] = None) -> list[str]:
    """맞은 술어 목록 (비면 통과). history는 다시 만든 턴까지 포함한 전체, new는 다시 만든 상담원 응답들."""
    texts = [r.text for r in new]
    tools = [c.name for r in new for c in r.tool_calls]
    hits = []
    exchanges, asked = [], ""
    for role, text in history:
        if role == "customer":
            asked = text
        else:
            exchanges.append((asked, text))
    first_new = len(exchanges) - len(new)
    for pred in fail_if:
        (k, v), = pred.items()
        hit = {
            "len_over": lambda: any(len(t) > int(v) for t in texts),
            "repeat": lambda: any(i >= max(first_new, 1) and _similar(exchanges[i - 1][1], a) and not _similar(exchanges[i - 1][0], q)
                                  for i, (q, a) in enumerate(exchanges)),
            "regex": lambda: any(re.search(v, t) for t in texts),
            "tool": lambda: v in tools,
            "no_tool": lambda: v not in tools,
            "event": lambda: any(e["kind"] == v for e in events),
            "gate": lambda: any(e["kind"] == "gate" and e["data"].get("reason") == v for e in events),
            "disposition": lambda: disposition == v,
            "check": lambda: failed_checks is not None and v in failed_checks,
        }.get(k)
        if hit is None:
            raise ReproError(f"모르는 술어: {k}")
        if k == "check" and failed_checks is None:
            raise ReproError("check: 는 continue 모드에서만 (통화 전체 채점)")
        if hit():
            hits.append(f"{k}: {v}")
    return hits


# -- 실행 --

def run_turn(agent: Any, stack: Any, sc: Scenario, row: dict[str, Any], spec: ReproSpec, *,
             customer: Any = None, judge: Any = None) -> dict[str, Any]:
    """turn·continue 한 번. customer가 있으면(continue) 그 턴 뒤를 시뮬레이터로 이어 간다."""
    turn, _ = spec.coordinate()
    state, utterance, ev = restore(stack, sc, row, turn)
    if spec.mode == "continue" and spec.customer == "recorded":
        later = [e["data"]["utterance"] for e in row["trace"] if e["kind"] == "turn_start" and e["turn"] > turn]
        customer = ScriptedCustomer(later)
    before = len(state.trace.events)
    try:
        reply = agent.respond(state, TurnInput(utterance, event=ev))
        tr = Transcript(sc.id, state, agent_name=getattr(agent, "name", ""), replies=_replies(row)[:turn - 1] + [reply])
        tr.disposition = reply.disposition or next((r.disposition for r in reversed(tr.replies) if r.disposition), None)
        failed = None
        if spec.mode == "continue" and not reply.end_call:
            _converse(agent, customer, replace(sc, max_turns=max(sc.max_turns - turn, 0)), tr)
        if spec.mode == "continue":
            failed = {c.name for c in score(sc, tr, stack.tools, judge).checks if c.passed is False}
        new = tr.replies[turn - 1:]
        events = state.trace.to_dicts()[before:]
        hits = judge_preds(spec.fail_if, new=new, history=state.history, events=events,
                           disposition=tr.disposition, failed_checks=failed)
        return {"failed": bool(hits), "hits": hits, "texts": [r.text for r in new],
                "tools": [[c.name for c in r.tool_calls] for r in new], "disposition": tr.disposition,
                "failed_checks": sorted(failed) if failed is not None else None, "trace": events,
                "call_id": state.call_id}
    finally:
        stack.close(state)


def run_step(llm: Any, source: Path, row: dict[str, Any], spec: ReproSpec) -> list[dict[str, Any]]:
    """그 스텝의 LLM 입력 원문을 그대로 repeat번 — 모델 출력의 흔들림만 본다."""
    turn, step = spec.coordinate()
    if step is None:
        raise ReproError("step 모드는 스텝까지 지정한다 (예: t1.s2)")
    llm_ev = next((e for e in coords(row["trace"]) if e["turn"] == turn and e["step"] == step and e["kind"] == "llm"),
                  None)
    if llm_ev is None or not llm_ev["data"].get("call_id"):
        raise ReproError(f"t{turn}.s{step}에 LLM 호출 기록이 없다")
    cid = llm_ev["data"]["call_id"]
    rec = next((json.loads(line) for line in (source / "llm_calls.jsonl").open(encoding="utf-8")
                if f'"{cid}"' in line and json.loads(line)["call_id"] == cid), None)
    if rec is None:
        raise ReproError(f"llm_calls.jsonl에 {cid} 없음")
    preds = [p for p in spec.fail_if if next(iter(p)) in ("len_over", "regex", "tool", "no_tool")]
    if len(preds) < len(spec.fail_if):
        raise ReproError("step 모드 술어는 len_over·regex·tool·no_tool만 (출력 한 번만 보므로)")
    out = []
    for _ in range(spec.repeat):
        res = llm.chat(rec["messages"], **rec["params"])
        try:
            obj = json.loads(res.text)
        except ValueError:
            obj = {}
        say, tool = obj.get("say") or "", (obj.get("tool") or {}).get("name")
        reply = AgentReply(text=say, tool_calls=[ToolCall(tool, {})] if tool else [])
        hits = judge_preds(preds, new=[reply], history=[("agent", say)], events=[])
        out.append({"failed": bool(hits), "hits": hits, "texts": [say], "tools": [[tool] if tool else []],
                    "output": res.text})
    return out
