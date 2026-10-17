"""저장된 실행 결과를 현재 시나리오 기대값으로 재채점하고, 여러 실행을 나란히 비교한다.

    python scripts/rescore.py runs/A runs/B [--dir scenarios scenarios_heldout]

다시 계산하는 것: 경로 채점(milestones · critical_exec, csagent/evaluation/path.py)과
말하기 진단(speech_diagnostics, 통과 판정 제외) — 응답 텍스트·trace만 보면 되는 항목.
LLM Judge 항목과 나머지 규칙 항목은 저장된 판정을 그대로 쓴다 (재실행·재판정 없음).
그래서 기준선 등 이전 실행에도 같은 기준을 적용해 비교할 수 있다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.config import ROOT
from csagent.evaluation.path import check_path, tool_stats
from csagent.evaluation.scenario import load_scenarios
from csagent.evaluation.scoring import MAX_REPLY_CHARS, MAX_UTTERANCE_CHARS, path_checks, speech_diagnostics

_RECOMPUTED = {"milestones", "critical_exec"}


def rescore(run_dir: Path, expect: dict) -> list[dict]:
    rows = []
    for line in (run_dir / "results.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        sc = expect.get(r["scenario_id"])
        if sc is None:
            continue
        checks = [c for c in r["checks"] if c["name"] not in _RECOMPUTED]
        checks += [c.__dict__ for c in path_checks(sc.milestones, r["trace"])]
        path = check_path(sc.milestones, r["trace"]) if sc.milestones else None
        rows.append({**r, "checks": checks, "passed": all(c["passed"] is not False for c in checks),
                     "path": path, "tools": tool_stats(r["trace"]),
                     "speech": {c.name: c for c in speech_diagnostics([x["text"] for x in r["replies"]], r["trace"])}})
    return rows


def report(runs: list[tuple[str, list[dict]]]) -> str:
    out = []
    names = [n for n, _ in runs]
    w = max(12, *(len(n) for n in names))

    # -- 전체 --
    out.append(f"{'':<22}" + "".join(f"{n:>{w + 2}}" for n in names))
    def line(label, fn):
        out.append(f"{label:<22}" + "".join(f"{fn(rows):>{w + 2}}" for _, rows in runs))
    pct = lambda a, b: f"{a}/{b} ({a / b:.0%})" if b else "-"
    line("통과", lambda rows: pct(sum(r["passed"] for r in rows), len(rows)))
    line("지점 완주", lambda rows: pct(sum(r["path"].passed for r in rows if r["path"]),
                                    sum(1 for r in rows if r["path"])))
    line("지점 진행도(평균)", lambda rows: (lambda ps: f"{sum(p.reached / p.total for p in ps) / len(ps):.0%}"
                                      if ps else "-")([r["path"] for r in rows if r["path"]]))
    line("치명적 실행 (통화 수)", lambda rows: str(sum(1 for r in rows if any(
        c["name"] == "critical_exec" and c["passed"] is False for c in r["checks"]))))
    line(f"턴 {MAX_REPLY_CHARS}자 초과 (통화 수)", lambda rows: str(sum(1 for r in rows if any(
        len(x["text"]) > MAX_REPLY_CHARS for x in r["replies"]))))
    for key, label in (("questions_per_turn", "질문 2개+ 턴 (통화 수)"), ("utterance_len", f"발화 {MAX_UTTERANCE_CHARS}자 초과 (통화 수)"),
                       ("spoken_unsaid", "버려진 말 노출 (통화 수)")):
        line(label, lambda rows, key=key: "-" if all(r["speech"][key].passed is None for r in rows)
             else str(sum(1 for r in rows if r["speech"][key].passed is False)))
    tool_keys = sorted({k for _, rows in runs for r in rows for k in r["tools"]})
    for k in tool_keys:
        line(f"  {k}", lambda rows, k=k: str(sum(r["tools"][k] for r in rows)))

    # -- 시나리오별: 통과 n/m · 지점 (끊긴 지점 최빈) --
    out.append("")
    out.append(f"{'시나리오':<26}" + "".join(f"{n:>{w + 2}}" for n in names))
    by = [defaultdict(list) for _ in runs]
    for i, (_, rows) in enumerate(runs):
        for r in rows:
            by[i][r["scenario_id"]].append(r)
    sids = list(dict.fromkeys(sid for b in by for sid in b))
    breaks = defaultdict(Counter)
    for sid in sids:
        cells = []
        for i, b in enumerate(by):
            rs = b.get(sid, [])
            cell = f"{sum(r['passed'] for r in rs)}/{len(rs)}" if rs else "-"
            ps = [r["path"] for r in rs if r["path"]]
            if ps:
                cell += f" 지점{sum(p.passed for p in ps)}/{len(ps)}"
                for p in ps:
                    if p.broken_at:
                        breaks[sid][(names[i], p.broken_at)] += 1
            cells.append(cell)
        out.append(f"{sid:<26}" + "".join(f"{c:>{w + 2}}" for c in cells))

    if breaks:
        out.append("\n끊긴 지점")
        for sid, c in breaks.items():
            out.append(f"  {sid}: " + ", ".join(f"[{n}] {m} ×{k}" for (n, m), k in c.most_common()))

    crit = [(n, r["scenario_id"], r["repeat"], c["detail"]) for n, rows in runs for r in rows
            for c in r["checks"] if c["name"] == "critical_exec" and c["passed"] is False]
    if crit:
        out.append("\n치명적 실행")
        out += [f"  [{n}] {sid} #{rep}: {d}" for n, sid, rep, d in crit]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="실행 폴더 (results.jsonl)")
    ap.add_argument("--dir", nargs="+", default=[str(ROOT / "scenarios"), str(ROOT / "scenarios_heldout")])
    args = ap.parse_args()
    expect = {s.id: s.expect for d in args.dir for s in load_scenarios(Path(d))}
    runs = [(Path(p).name, rescore(Path(p), expect)) for p in args.runs]
    print(report(runs))


if __name__ == "__main__":
    main()
