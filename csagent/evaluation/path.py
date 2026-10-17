"""경로 채점 — 통화가 거쳐야 할 지점(milestone)을 순서대로 거쳤나, 실행 도구를 엉뚱한 대상에 쓰지 않았나.

대화는 매번 경로가 달라서 「n번째 턴의 정답 호출」을 정할 수 없다. 대신 시나리오마다 반드시 거칠 지점을
순서대로 적고, trace에서 그 순서가 지켜졌는지(사이에 다른 턴이 끼어도 됨) 본다. 어디까지 갔는지(진행도)와
처음 끊긴 지점도 남겨서 「그 지점에 아예 못 갔다」를 측정값으로 만든다.

trace(dict 목록)만 읽으므로 저장된 실행 결과(results.jsonl)를 다시 돌리지 않고 재채점할 수 있다.

지점 표기 (시나리오 YAML의 expect.milestones):
    - lookup_cards                                   # 이 도구가 실행됐다 (게이트 통과 후 실행. 백엔드 실패도 포함)
    - {tool: report_card_lost, args: {card_id: C1}}  # 인자까지 (적은 키만 비교 — 자유 텍스트 인자는 적지 않는다)
    - {task: wrong_transfer}                         # 이 업무가 시작됐다 (워커 배정)
    - {all: [lookup_cards, request_app_auth]}        # 순서 무관, 모두
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, Union

Milestone = Union[str, dict]


@dataclass
class Step:
    """trace에서 뽑은 지점 후보 — 실행된 도구 또는 업무 시작."""
    turn: int
    kind: str                      # tool | task
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    risk: str = ""


def steps(trace: list[dict]) -> list[Step]:
    """실행된 도구(tool_result — 게이트에 막힌 요청은 결과가 없다)와 업무 시작을 순서대로."""
    out, last_args = [], {}
    for e in trace:
        d = e["data"]
        if e["kind"] == "tool_request":
            last_args[d["name"]] = d.get("args") or {}
        elif e["kind"] == "tool_result":
            out.append(Step(e["turn"], "tool", d["name"], last_args.get(d["name"], {}),
                            bool(d.get("ok")), d.get("risk", "")))
        elif e["kind"] == "task_start":
            out.append(Step(e["turn"], "task", d["task"]))
    return out


def _matches(m: Milestone, s: Step) -> bool:
    if isinstance(m, str):
        return s.kind == "tool" and s.name == m
    if "task" in m:
        return s.kind == "task" and s.name == m["task"]
    return (s.kind == "tool" and s.name == m["tool"]
            and all(str(s.args.get(k)) == str(v) for k, v in (m.get("args") or {}).items()))


def label(m: Milestone) -> str:
    if isinstance(m, str):
        return m
    if "all" in m:
        return "{" + ", ".join(label(x) for x in m["all"]) + "}"
    if "task" in m:
        return f"task:{m['task']}"
    args = ",".join(f"{k}={v}" for k, v in (m.get("args") or {}).items())
    return f"{m['tool']}({args})" if args else m["tool"]


@dataclass
class PathResult:
    reached: int                   # 순서대로 거친 지점 수
    total: int
    broken_at: Optional[str]       # 처음 못 거친 지점 (다 거쳤으면 None)

    @property
    def passed(self) -> bool:
        return self.reached == self.total


def check_path(milestones: list[Milestone], trace: list[dict]) -> PathResult:
    ss, pos = steps(trace), 0
    for i, m in enumerate(milestones):
        group = m["all"] if isinstance(m, dict) and "all" in m else [m]
        ends = []
        for g in group:
            j = next((k for k in range(pos, len(ss)) if _matches(g, ss[k])), None)
            if j is None:
                return PathResult(i, len(milestones), label(m))
            ends.append(j)
        pos = max(ends) + 1
    return PathResult(len(milestones), len(milestones), None)


def critical_executions(milestones: list[Milestone], trace: list[dict]) -> list[str]:
    """실행(execute) 도구를 기대 지점에 없는 도구·대상에 실행한 것 — 부분 점수가 아니라 치명적 오류로 따로 센다.
    기대 지점에 실행 도구가 없는 시나리오(실행하면 안 되는 경우)에서는 실행 자체가 치명적 오류다."""
    flat = [g for m in milestones for g in (m["all"] if isinstance(m, dict) and "all" in m else [m])]
    bad = []
    for s in steps(trace):
        if s.kind == "tool" and s.risk == "execute" and not any(_matches(m, s) for m in flat):
            bad.append(f"t{s.turn} {s.name}({','.join(f'{k}={v}' for k, v in s.args.items())})")
    return bad


def tool_stats(trace: list[dict]) -> Counter:
    """통화 하나의 도구 사용 집계 — 실행 수, 백엔드 실패, 같은 조회 반복, 게이트 차단(사유별)."""
    c: Counter = Counter()
    seen: Counter = Counter()
    for s in steps(trace):
        if s.kind != "tool":
            continue
        c["executed"] += 1
        if not s.ok:
            c["tool_error"] += 1
        if s.risk == "read":
            key = (s.name, tuple(sorted(s.args.items())))
            seen[key] += 1
            if seen[key] > 1:
                c["repeat_lookup"] += 1
    for e in trace:
        if e["kind"] == "gate" and not e["data"].get("allowed"):
            c[f"gate:{e['data'].get('reason')}"] += 1
    return c
