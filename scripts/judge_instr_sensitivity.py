"""근거 판정이 워커 지시문 문구에 흔들리는지 잰다 — 같은 통화·같은 발화·같은 검색 문서·같은 도구 결과에 지시문만 바꿔 다시 판정.

    python scripts/judge_instr_sensitivity.py runs/A [runs/B] --instr old=labels/instructions_995c0ab.json [--k 2]

지시문 변형: current(지금 코드의 워커 지시문) · none(지시문 없음) · 이름=JSON 파일(워커 이름 → 지시문)
에이전트는 다시 돌리지 않는다 (results.jsonl의 trace로 판정 입력을 다시 만든다, scripts/rejudge.py와 같은 방식).
판정기가 결정적이면(같은 입력 k번 → 같은 판정) 변형 사이의 뒤집힘은 전부 지시문 때문이다.

출력 (labels/instr_sens_<시각>/): verdicts.jsonl — 턴 하나 = {key, turn, say, verdicts: {변형: [통과?…]}, claims}
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.agent.prompts import BASELINE_TASK
from csagent.agent.workers.specs import WORKERS
from csagent.config import ROOT
from csagent.evaluation.scoring import LLMJudge
from csagent.factory import build_llm
from scripts.rejudge import saved_transcript


def load_variants(specs: list[str]) -> dict[str, dict[str, str]]:
    out = {"current": {"baseline": BASELINE_TASK, **{n: w.policy.task for n, w in WORKERS.items()}}, "none": {}}
    for s in specs:
        name, path = s.split("=", 1)
        out[name] = json.loads(Path(path).read_text(encoding="utf-8"))
    return out


def judge_turns(llm, variants: dict[str, dict[str, str]], row: dict, k: int, grounding: str = "v4") -> list[dict]:
    tr = saved_transcript(row)
    key = f"{row['_run']}/{row['scenario_id']}/r{row['repeat']}"
    per_turn: dict[int, dict] = {}
    for name, instr in variants.items():
        judge = LLMJudge(llm, instr, grounding=grounding)
        for _ in range(k):
            for g in judge.grounded_turns(tr):
                t = per_turn.setdefault(g["turn"], {"key": key, "scenario_id": row["scenario_id"], "turn": g["turn"],
                                                    "say": tr.replies[g["turn"] - 1].text,
                                                    "verdicts": {}, "claims": {}})
                passed = None if g["unsupported"] is None else not g["unsupported"]
                t["verdicts"].setdefault(name, []).append(passed)
                if g["unsupported"]:
                    t["claims"].setdefault(name, []).append(g["claim"])
                if grounding == "v4":   # 주장별 판정 (변형 사이에 주장 목록이 같은지·어느 주장이 뒤집혔는지)
                    t.setdefault("v4", {}).setdefault(name, []).append(
                        [{k2: c.get(k2) for k2 in ("span", "verdict", "source", "evidence", "note")}
                         for c in g.get("claims") or []])
    return list(per_turn.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--instr", nargs="*", default=[], help="이름=지시문 JSON (워커 이름 → 지시문)")
    ap.add_argument("--k", type=int, default=2, help="변형마다 같은 입력 재판정 횟수 (판정기 결정성 확인)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--grounding", default="v4", help="근거 판정 방식 v4(2단계) | v3")
    args = ap.parse_args()

    variants = load_variants(args.instr)
    rows = [{**json.loads(line), "_run": Path(run).name}
            for run in args.runs for line in (Path(run) / "results.jsonl").open(encoding="utf-8")]
    out_dir = ROOT / "labels" / f"instr_sens_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True)
    print(f"통화 {len(rows)}건 · 변형 {list(variants)} · k={args.k} · 판정 {args.grounding} → {out_dir.relative_to(ROOT)}",
          flush=True)

    llm = build_llm()
    errors: list[dict] = []

    def safe(r: dict) -> list[dict]:
        # LLM 재시도(csagent/llm.py)까지 실패한 통화는 건너뛰고 기록만 남긴다 — 하나 때문에 전체를 버리지 않게
        try:
            return judge_turns(llm, variants, r, args.k, args.grounding)
        except Exception as e:
            errors.append({"key": f"{r['_run']}/{r['scenario_id']}/r{r['repeat']}", "error": repr(e)[:300]})
            print(f"[ERR] {errors[-1]['key']} {type(e).__name__}", flush=True)
            return []

    with ThreadPoolExecutor(args.workers) as ex:
        turns = [t for ts in ex.map(safe, rows) for t in ts]
    if errors:
        (out_dir / "errors.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in errors),
                                              encoding="utf-8")
        print(f"!! 건너뛴 통화 {len(errors)}건 (errors.jsonl)", flush=True)
    with (out_dir / "verdicts.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in turns)
    (out_dir / "variants.json").write_text(json.dumps({"runs": args.runs, "instr": args.instr, "k": args.k,
                                                       "grounding": args.grounding},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")

    names = list(variants)
    print(f"\n턴 {len(turns)}개")
    for n in names:
        vs = [t["verdicts"][n] for t in turns]
        unstable = sum(len({x for x in v if x is not None}) > 1 for v in vs)
        fails = sum(v[0] is False for v in vs)
        print(f"  [{n:8}] 근거 없음 판정 {fails}턴 · 같은 입력 재판정 흔들림 {unstable}턴 · 파싱 실패 {sum(None in v for v in vs)}턴")
    print("\n변형 사이 뒤집힘 (첫 판정 기준)")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            flips = [t for t in turns if None not in (t["verdicts"][a][0], t["verdicts"][b][0])
                     and t["verdicts"][a][0] != t["verdicts"][b][0]]
            dirs = Counter(f"{a}:{'통과' if t['verdicts'][a][0] else '실패'}→{b}:{'통과' if t['verdicts'][b][0] else '실패'}"
                           for t in flips)
            print(f"  {a} vs {b}: {len(flips)}턴  {dict(dirs)}")
    print(f"\n자세히: {out_dir.relative_to(ROOT)}/verdicts.jsonl")


if __name__ == "__main__":
    main()
