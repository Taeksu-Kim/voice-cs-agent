"""저장된 실행을 「고객이 실제로 들은 말」로 다시 구성해, 지금 판정기·지금 시나리오 기대값으로 똑같이 다시 채점한다.

    python scripts/rescore_heard.py runs/A runs/B [--only s1_ s4_] [--skip s3_] [--workers 2]

왜: 10/19부터 기록·채점이 들은 말 기준이다 (docs/09 G3). 그 전 실행은 막히거나 실패한 스텝의 말이 기록에서 빠져 있어
점수를 직접 비교할 수 없다. 에이전트는 다시 돌리지 않는다 — trace의 spoken(스트리밍으로 나간 말)으로 턴별 응답을 다시 만들고,
코드가 덧붙인 말(완료 한마디·대체 문장)은 spoken에 안 남으므로 응답 텍스트에서 보탠다.
두 실행 모두 같은 함수로 다시 구성하고 같은 판정기로 채점한다 (판정기 잡음은 두 쪽에 똑같이 들어간다).

출력: labels/rescore_heard_<시각>/results.jsonl (실행·시나리오·반복별 통과·실패 항목), 화면에 비교 표
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.agent.contracts import AgentReply, ToolCall
from csagent.agent.loop import FALLBACK, TOOL_STUCK
from csagent.agent.prompts import BASELINE_TASK
from csagent.agent.state import CallState
from csagent.agent.trace import Trace, TraceEvent
from csagent.agent.workers.specs import WORKERS
from csagent.config import ROOT
from csagent.evaluation.runner import Transcript
from csagent.evaluation.scenario import load_scenarios
from csagent.evaluation.scoring import LLMJudge, score
from csagent.factory import build_llm
from csagent.stack import offline_registry

# 코드가 바로 붙이는 말 — spoken trace에 안 남는다 (loop.py done_line·대체 문장)
CODE_LINES = (FALLBACK, TOOL_STUCK, "보내 드렸어요.", "예약해 두었어요.")


def heard_text(turn_events: list[dict], reply_text: str) -> str:
    spoken = " ".join(e["data"]["text"].strip() for e in turn_events if e["kind"] == "spoken")
    extra = [x for x in CODE_LINES if x in reply_text and x not in spoken]
    return " ".join(x for x in [spoken, *extra] if x)


def rebuild(row: dict, sc) -> Transcript:
    events = [TraceEvent(**{k: e[k] for k in ("turn", "kind", "data", "t")}) for e in row["trace"]]
    by_turn: dict[int, list[dict]] = defaultdict(list)
    for e in row["trace"]:
        by_turn[e["turn"]].append(e)
    replies, heard = [], []
    for i, r in enumerate(row["replies"], 1):
        text = heard_text(by_turn[i], r["text"])
        heard.append(text)
        replies.append(AgentReply(text=text, retrieved=r.get("retrieved") or [],
                                  tool_calls=[ToolCall(n, a, res) for n, a, res in r.get("tools") or []]))
    history, k = [], 0
    for role, text in row["turns"]:
        if role == "agent" and history and history[-1][0] == "customer":   # 고객 말 뒤 에이전트 응답 = 턴 k
            text = heard[k] if k < len(heard) else text
            k += 1
        history.append((role, text))
    state = CallState(ctx=sc.context)
    state.history, state.trace = history, Trace(events)
    return Transcript(row["scenario_id"], state, agent_name=row.get("agent", ""), replies=replies,
                      disposition=row["disposition"], end_reason=row.get("end_reason", ""),
                      customer_satisfied=row.get("customer_satisfied"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--only", nargs="*", default=[""])
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--dir", nargs="+", default=[str(ROOT / "scenarios")])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    scenarios = {s.id: s for d in args.dir for s in load_scenarios(Path(d))}
    instructions = {"baseline": BASELINE_TASK, **{n: w.policy.task for n, w in WORKERS.items()}}
    judge = LLMJudge(build_llm(source="judge"), instructions)
    tools = offline_registry()

    jobs = []
    for run in args.runs:
        for line in (Path(run) / "results.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            sid = row["scenario_id"]
            if sid in scenarios and sid.startswith(tuple(args.only)) and not sid.startswith(tuple(args.skip) or ("\0",)):
                jobs.append((Path(run).name, row))

    def work(job):
        run, row = job
        sc = scenarios[row["scenario_id"]]
        sco = score(sc, rebuild(row, sc), tools, judge)
        return {"run": run, "scenario_id": sc.id, "repeat": row["repeat"], "passed": sco.passed,
                "original_passed": row["passed"],
                "failed": [f"{c.name}: {(c.detail or '')[:120]}" for c in sco.checks if c.passed is False]}

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(work, jobs))

    out = ROOT / "labels" / f"rescore_heard_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    runs = [Path(r).name for r in args.runs]
    table: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        table[r["scenario_id"]][r["run"]].append(r["passed"])
    print(f"\n{'시나리오':28s} " + "  ".join(f"{r[:15]:>15s}" for r in runs))
    for sid in sorted(table):
        print(f"{sid:28s} " + "  ".join(f"{sum(table[sid][r])}/{len(table[sid][r]):>13}" for r in runs))
    for r in runs:
        rows = [x for x in results if x["run"] == r]
        print(f"{r}: 들은 말 재채점 {sum(x['passed'] for x in rows)}/{len(rows)} "
              f"(저장된 원래 점수 {sum(bool(x['original_passed']) for x in rows)}/{len(rows)})")
    print(f"결과 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
