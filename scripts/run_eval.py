"""평가 실행: 시나리오 전부를 고객 시뮬레이터 ↔ 에이전트로 돌리고 채점한다.

    python scripts/run_eval.py [--only s1_] [--no-judge] [--repeat 1]

결과: runs/<시각>/results.jsonl (통화 기록 + 채점), 화면에는 요약 표.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.config import ROOT
from csagent.factory import build_agent, build_llm
from csagent.evaluation.runner import run_scenario
from csagent.evaluation.scenario import load_scenarios
from csagent.evaluation.scoring import LLMJudge, score
from csagent.evaluation.simulator import LLMCustomer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="시나리오 id 접두어")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--agent", default="A1", help="A0 | A1")
    args = ap.parse_args()

    llm = build_llm()
    agent = build_agent(args.agent)
    judge = None if args.no_judge else LLMJudge(llm)
    scenarios = [x for x in load_scenarios(ROOT / "scenarios") if x.id.startswith(args.only)]

    out_dir = ROOT / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{agent.name}"
    out_dir.mkdir(parents=True)
    rows = []
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for sc in scenarios:
            for rep in range(args.repeat):
                tr = run_scenario(agent, LLMCustomer(llm), sc)
                sco = score(sc, tr, judge)
                rows.append((sc, tr, sco))
                f.write(json.dumps({"repeat": rep, **tr.to_dict(),
                                    "passed": sco.passed,
                                    "checks": [c.__dict__ for c in sco.checks]},
                                   ensure_ascii=False) + "\n")
                _print_row(sc.id, tr, sco)

    passed = sum(sco.passed for _, _, sco in rows)
    lat = [r.timings_ms["total"] for _, tr, _ in rows for r in tr.replies]
    print(f"\n통과 {passed}/{len(rows)} | 턴 응답 지연 중앙값 {st.median(lat):.0f}ms "
          f"p90 {sorted(lat)[int(len(lat) * .9)]:.0f}ms | 결과 {out_dir.relative_to(ROOT)}")


def _print_row(sid: str, tr, sco) -> None:
    mark = "PASS" if sco.passed else "FAIL"
    print(f"[{mark}] {sid}  턴 {len(tr.replies)}  결과 {tr.disposition}  ({tr.end_reason})")
    for c in sco.failed():
        print(f"    ✗ {c.name}: {c.detail}")


if __name__ == "__main__":
    main()
