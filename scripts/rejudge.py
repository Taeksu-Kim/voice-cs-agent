"""저장된 실행의 LLM Judge 판정을 같은 입력으로 여러 번 다시 돌려 판정 일관성을 잰다 (1-7 Judge 신뢰도).

    python scripts/rejudge.py runs/A runs/B [--k 3] [--shuffle 1] [--dir scenarios scenarios_heldout]

에이전트는 다시 돌리지 않는다 — results.jsonl의 통화 기록·trace로 판정 입력을 그대로 다시 만든다.
- 통화 판정(mention·must_not·goal_resolved·false_completion): 원래 순서 k번 + 질문 순서를 섞어 shuffle번
- 근거 판정(grounded): 턴 단위로 k번 (원래 기록은 통화 단위로 합쳐져 있어, 실패 턴만 detail에서 복원)

출력 (labels/rejudge_<시각>/):
  verdicts.jsonl  항목 하나 = {key, run, scenario_id, repeat, item, kind, original, verdicts[]}
                  original·verdicts 값은 「통과했는가」 (True/False, 파싱 실패 None)
  items.jsonl     항목별 판정 입력 (라벨 붙이기용 — 판정 결과는 넣지 않는다)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.agent.prompts import BASELINE_TASK
from csagent.agent.tools.mock import default_registry
from csagent.agent.trace import Trace, TraceEvent
from csagent.agent.workers.specs import WORKERS
from csagent.config import ROOT
from csagent.evaluation.procedure import audit
from csagent.evaluation.scenario import load_scenarios
from csagent.evaluation.scoring import LLMJudge
from csagent.factory import build_llm

_FAILED_TURN = re.compile(r"(\d+)턴:")


def item_kind(item: str) -> str:
    return re.sub(r"_t?\d+$", "", item)


def saved_transcript(row: dict) -> SimpleNamespace:
    """판정기가 쓰는 부분(turns · disposition · replies[].text · state.trace)만 복원한다."""
    trace = Trace([TraceEvent(**{k: e[k] for k in ("turn", "kind", "data", "t")}) for e in row["trace"]])
    return SimpleNamespace(turns=[tuple(t) for t in row["turns"]], disposition=row["disposition"],
                           replies=[SimpleNamespace(text=r["text"]) for r in row["replies"]],
                           state=SimpleNamespace(trace=trace))


def original_verdicts(row: dict, n_turns: int) -> dict[str, bool | None]:
    out = {c["name"]: c["passed"] for c in row["checks"]}
    g = next((c for c in row["checks"] if c["name"] == "grounded"), None)
    if g is not None:
        failed = {int(t) for t in _FAILED_TURN.findall(g["detail"] or "")} if g["passed"] is False else set()
        for t in range(1, n_turns + 1):
            out[f"grounded_t{t}"] = None if g["passed"] is None else t not in failed
    return out


def judge_row(llm, instructions, sc, row: dict, k: int, shuffle: int) -> tuple[list[dict], list[dict]]:
    tr = saved_transcript(row)
    proc = audit(tr.state.trace, default_registry())
    verdicts: dict[str, list] = defaultdict(list)
    for v in range(k + shuffle):
        judge = LLMJudge(llm, instructions, order_seed=None if v < k else v)
        for c in judge._conversation_checks(sc, tr, proc):
            verdicts[c.name].append(c.passed)
        if v < k:
            for g in judge.grounded_turns(tr):
                verdicts[f"grounded_t{g['turn']}"].append(None if g["unsupported"] is None else not g["unsupported"])
    orig = original_verdicts(row, len(tr.replies))
    base = {"run": row["_run"], "scenario_id": row["scenario_id"], "repeat": row["repeat"]}
    key = f"{row['_run']}/{row['scenario_id']}/r{row['repeat']}"

    judge = LLMJudge(llm, instructions)
    questions = {qid: (q, want_yes) for qid, q, want_yes in judge.conversation_questions(sc, tr, proc)}
    grounded = {f"grounded_t{g['turn']}": g for g in judge.grounded_inputs(tr)}
    vrows, irows = [], []
    for item, vs in verdicts.items():
        vrows.append({"key": f"{key}/{item}", **base, "item": item, "kind": item_kind(item),
                      "original": orig.get(item), "verdicts": vs})
        inp = {"key": f"{key}/{item}", **base, "item": item, "kind": item_kind(item),
               "goal": sc.customer.get("goal", ""), "transcript": judge.transcript_text(tr)}
        if item in questions:
            q, want_yes = questions[item]
            inp.update(question=q, pass_if="YES" if want_yes else "NO")
        else:
            g = grounded[item]
            inp.update(turn=g["turn"], say=g["say"], docs=g["docs"], tools=g["tools"],
                       instructions=g["instructions"], pass_if="근거 있음")
        irows.append(inp)
    return vrows, irows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--k", type=int, default=3, help="원래 순서 재판정 횟수")
    ap.add_argument("--shuffle", type=int, default=1, help="질문 순서를 섞은 재판정 횟수 (통화 판정만)")
    ap.add_argument("--dir", nargs="+", default=[str(ROOT / "scenarios"), str(ROOT / "scenarios_heldout")])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    scenarios = {s.id: s for d in args.dir for s in load_scenarios(Path(d))}
    rows = []
    for run in args.runs:
        for line in (Path(run) / "results.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if r["scenario_id"] in scenarios:
                rows.append({**r, "_run": Path(run).name})
    out_dir = ROOT / "labels" / f"rejudge_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True)
    print(f"통화 {len(rows)}건 → {out_dir.relative_to(ROOT)}", flush=True)

    llm = build_llm()
    instructions = {"baseline": BASELINE_TASK, **{n: w.policy.task for n, w in WORKERS.items()}}
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda r: judge_row(llm, instructions, scenarios[r["scenario_id"]], r,
                                                   args.k, args.shuffle), rows))
    vrows = [v for vs, _ in results for v in vs]
    with (out_dir / "verdicts.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(v, ensure_ascii=False) + "\n" for v in vrows)
    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(i, ensure_ascii=False) + "\n" for _, irows in results for i in irows)

    # 요약: 종류별 재판정 뒤집힘 (원래 순서끼리 / 순서 섞기 포함 / 저장된 원래 판정과)
    stats = defaultdict(Counter)
    for v in vrows:
        s, vs = stats[v["kind"]], v["verdicts"]
        same = vs[:args.k]
        s["items"] += 1
        s["parse_fail"] += any(x is None for x in vs)
        s["flip_k"] += len({x for x in same if x is not None}) > 1
        s["flip_all"] += len({x for x in vs if x is not None}) > 1
        s["vs_original"] += v["original"] is not None and any(x is not None and x != v["original"] for x in vs)
        s["fail_majority"] += sum(x is False for x in same) * 2 > len(same)
    print(f"\n{'종류':<18}{'항목':>6}{'실패(다수결)':>12}{'k회 뒤집힘':>12}{'섞기 포함':>10}{'원판정과 다름':>14}{'파싱실패':>8}")
    for kind, s in sorted(stats.items()):
        print(f"{kind:<18}{s['items']:>6}{s['fail_majority']:>12}{s['flip_k']:>12}{s['flip_all']:>10}"
              f"{s['vs_original']:>14}{s['parse_fail']:>8}")


if __name__ == "__main__":
    main()
