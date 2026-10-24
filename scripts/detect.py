"""이상 탐지 실행 — 기록 폴더의 끝난 통화에서 신호를 찾고, 신호가 있는 통화마다 케이스 묶음을 만든다 (csagent/anomaly.py, docs/10 4절).

    python scripts/detect.py runs/20261020_071342_C
    python scripts/detect.py logs/20261024 --judge          # 서버 통화는 채점이 없다 — 근거 판정(v4)을 돌린다
    python scripts/detect.py runs/A runs/B --level bad      # bad 신호가 있는 통화만 묶음으로

결과: runs/cases/<시각>/ — summary.md(통화별 신호·가드레일 발동) · <출처>_<키>/(case.json · trace.md · llm_calls.jsonl · repro.yaml)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.anomaly import detect, write_case
from csagent.config import ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="runs/<실행> 또는 logs/<날짜>")
    ap.add_argument("--judge", action="store_true", help="서버 통화에 근거 판정(v4)")
    ap.add_argument("--level", choices=("bad", "warn"), default="warn", help="이 등급 이상 신호가 있는 통화만 묶음으로")
    ap.add_argument("--out")
    args = ap.parse_args()

    judge = None
    if args.judge:
        from csagent.agent.prompts import BASELINE_TASK
        from csagent.agent.workers.specs import WORKERS
        from csagent.evaluation.scoring import LLMJudge
        from csagent.factory import build_llm
        judge = LLMJudge(build_llm(source="judge"), {"baseline": BASELINE_TASK, **{n: w.policy.task for n, w in WORKERS.items()}})

    items = []
    for s in args.sources:
        src = Path(s) if Path(s).is_absolute() else ROOT / s
        items += detect(src, judge)
    levels = ("bad",) if args.level == "bad" else ("bad", "warn")
    cases = [it for it in items if any(sig.level in levels for sig in it["signals"])]
    cases.sort(key=lambda it: (-sum(s.level == "bad" for s in it["signals"]), -len(it["signals"])))

    out = Path(args.out) if args.out else ROOT / "runs" / "cases" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# 이상 탐지 {out.name}", "", f"출처: {', '.join(args.sources)} · 통화 {len(items)} · 케이스 {len(cases)}", "",
             "| 케이스 | 신호 | 가드레일 발동 |", "|---|---|---|"]
    for it in cases:
        d = write_case(out, ROOT, it)
        sigs = "<br>".join(f"**{s.level}** {s.kind} `{s.at or '-'}` {s.detail[:60]}" for s in it["signals"])
        guards = ", ".join(f"{k} {v}" for k, v in it["guards"].items()) or "-"
        lines.append(f"| [{it['key']}]({d.name}/trace.md) | {sigs} | {guards} |")

    kinds = Counter(s.kind for it in items for s in it["signals"])
    guards = Counter()
    for it in items:
        guards.update(it["guards"])
    lines += ["", "## 신호 종류", "", *[f"- {k}: {v}" for k, v in kinds.most_common()],
              "", "## 가드레일 발동 (전체 통화 — 지시문 개선 후보)", "", *[f"- {k}: {v}" for k, v in guards.most_common()]]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"통화 {len(items)} · 케이스 {len(cases)} → {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    for it in cases:
        print(f"  {it['key']:32s} " + " | ".join(f"{s.level} {s.kind} {s.at}".strip() for s in it["signals"]))
    print("신호:", dict(kinds.most_common()))
    print("가드레일:", dict(guards.most_common()))


if __name__ == "__main__":
    main()
