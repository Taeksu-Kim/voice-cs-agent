"""통화 trace → mermaid 흐름도. 노드 id가 좌표(trace.py의 at)라서 그림의 노드를 가리키면 곧 재현 시작점이다.

    t2_s1_gate  ↔  at "t2.s1.gate"  ↔  재현 명세 from: <call_id>@t2.s1

표시: 빨강(bad) = 오류·실패 (턴 예외, 도구 실패, 결과 모름), 주황(guard) = 가드레일이 막거나 고친 곳
(게이트 차단, 보류한 말 버림, 절차 강제, 번호·링크 가림, 약속 재촉). 좌표 없는 예전 trace는 스텝을 추정한다 (coords).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

SHOW = ("route", "task_start", "task_switch", "protocol_step", "pending_answer", "llm", "tool_request", "precheck",
        "slot_rule", "gate", "tool_result", "execution_unknown", "spoken", "unheard_drop", "step_enforced",
        "promise_nudge", "number_blocked", "link_blocked", "question_hold", "spoken_confirm", "task_done",
        "turn_error")
GUARD = ("unheard_drop", "step_enforced", "promise_nudge", "number_blocked", "link_blocked", "precheck",
         "question_hold")
BAD = ("turn_error", "execution_unknown")
_PRE_LLM = ("spoken", "unheard_drop")   # 스텝 안에서 llm 이벤트보다 먼저 기록되는 것 (스트리밍 중)


def coords(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """각 이벤트에 at·step이 있게 한다. 이미 있으면 그대로, 없으면(10/21 전 trace) llm 이벤트 순서로 추정."""
    if all(e.get("at") for e in events):
        return events
    out, seen = [dict(e) for e in events], defaultdict(int)
    by_turn: dict[int, list[dict]] = defaultdict(list)
    for e in out:
        by_turn[e["turn"]].append(e)
    for evs in by_turn.values():
        step, buffered = 0, []
        for e in evs:
            if e["kind"] in _PRE_LLM:
                buffered.append(e)
            elif e["kind"] == "llm":
                step += 1
                for b in buffered + [e]:
                    b["step"] = step
                buffered = []
            else:
                e["step"] = step
        for b in buffered:   # 마지막 llm 뒤에 나온 말 (늦게 내보낸 말 등)
            b["step"] = step
    for e in out:
        at = f"t{e['turn']}.s{e['step']}.{e['kind']}"
        seen[at] += 1
        e["at"] = at if seen[at] == 1 else f"{at}#{seen[at]}"
    return out


def severity(e: dict[str, Any]) -> Optional[str]:
    k, d = e["kind"], e["data"]
    if k in BAD or (k == "tool_result" and d.get("ok") is False):
        return "bad"
    if k in GUARD or (k == "gate" and not d.get("allowed", True)):
        return "guard"
    return None


def node_id(at: str) -> str:
    return at.replace(".", "_").replace("#", "_n")


def _label(e: dict[str, Any]) -> str:
    k, d = e["kind"], e["data"]
    text = {
        "route": lambda: f"route → {d.get('task')}",
        "task_start": lambda: f"업무 시작 {d.get('task')}" + (" (이어받음)" if d.get("resumed") else ""),
        "task_switch": lambda: f"업무 전환 {d.get('from')} → {d.get('to')}",
        "protocol_step": lambda: f"단계 {d.get('step')}",
        "pending_answer": lambda: f"고객 답 {d.get('answer')} {'✓' if d.get('verified') else '✗'}",
        "llm": lambda: f"LLM {d.get('ms', 0) / 1000:.1f}s" + (" tool먼저" if d.get("tool_first") else ""),
        "tool_request": lambda: f"요청 {d.get('name')}",
        "gate": lambda: f"gate {'✓' if d.get('allowed') else '✗ ' + str(d.get('reason'))} {d.get('name')}",
        "tool_result": lambda: f"{d.get('name')} {'✓' if d.get('ok') else '✗ ' + str(d.get('error') or d.get('code') or '')}",
        "spoken": lambda: ("말(늦게): " if d.get("late") else "말: ") + str(d.get("text", "")),
        "unheard_drop": lambda: "버린 말: " + str(d.get("text", "")),
        "turn_error": lambda: "턴 오류 " + str(d.get("error", "")),
    }.get(k, lambda: k)()
    text = text.replace('"', "'").replace("\n", " ")
    return text if len(text) <= 48 else text[:47] + "…"


def to_mermaid(events: list[dict[str, Any]], title: str = "", failed: list[str] = ()) -> str:
    """failed: 채점 실패 항목 (평가 기록이면) — 그림 위 메모 노드로."""
    events = coords(events)
    lines = ["flowchart TD"]
    if title or failed:
        note = title + ("<br/>실패: " + "<br/>".join(f.replace('"', "'")[:70] for f in failed) if failed else "")
        lines.append(f'  head["{note}"]')
    by_turn: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        by_turn[e["turn"]].append(e)
    prev, marks = "head" if (title or failed) else None, defaultdict(list)
    for turn in sorted(t for t in by_turn if t > 0):
        evs = by_turn[turn]
        said = next((e["data"].get("utterance", "") for e in evs if e["kind"] == "turn_start"), "")
        wheres = list(dict.fromkeys(e.get("where") for e in evs if e.get("where")))
        head = f"t{turn} · {' → '.join(wheres)}" if wheres else f"t{turn}"
        said = said.replace('"', "'")
        lines.append(f'  subgraph T{turn}["{head}<br/>고객: {said[:40]}"]')
        lines.append("    direction LR")
        for e in evs:
            if e["kind"] not in SHOW:
                continue
            nid = node_id(e["at"])
            lines.append(f'    {nid}["s{e["step"]} {_label(e)}"]')
            if sev := severity(e):
                marks[sev].append(nid)
            if prev:
                lines.append(f"    {prev} --> {nid}")
            prev = nid
        lines.append("  end")
    lines += ["  classDef bad fill:#fdd,stroke:#c00,color:#600",
              "  classDef guard fill:#fec,stroke:#c80,color:#630"]
    lines += [f"  class {','.join(ids)} {sev}" for sev, ids in marks.items()]
    return "\n".join(lines) + "\n"
