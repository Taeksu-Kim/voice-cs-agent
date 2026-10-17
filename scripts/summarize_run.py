"""평가 실행 결과 요약 — 시나리오별 통과율, 실패 항목, 시뮬레이터 이탈, LLM 호출·토큰·지연.

    python scripts/summarize_run.py runs/20261013_012025_C [runs/... 여러 개면 합쳐서]

반복 실행(--repeat)의 통과율로 본다. 한 번 통과·실패로는 개선 여부를 판단하지 않는다.
시뮬레이터가 설정에 없는 사실을 지어낸 실행(sim_faithful 실패)은 따로 센다 — 그 실패는 측정 잡음일 수 있다.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.monitoring import aggregate


ERRORS_FILE = "errors.jsonl"   # 오류(LLM 타임아웃 등)로 건너뛴 시나리오 — 통과율 분모에 넣지 않는다


def summarize(run_dirs: list[Path]) -> str:
    rows = [json.loads(line) for d in run_dirs for line in (d / "results.jsonl").open(encoding="utf-8")]
    errs = [json.loads(line) for d in run_dirs if (d / ERRORS_FILE).exists()
            for line in (d / ERRORS_FILE).open(encoding="utf-8")]
    out = []
    if errs:
        out.append(f"오류로 건너뜀 {len(errs)}건: " + ", ".join(f"{e['scenario_id']}#{e['repeat']}" for e in errs))
    if not rows:
        return "\n".join(out + ["채점된 결과 없음"])

    # -- 시나리오별 통과율 --
    by_sc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sc[r["scenario_id"]].append(r)
    total_pass = sum(r["passed"] for r in rows)
    out.append(f"통과 {total_pass}/{len(rows)} ({total_pass / len(rows):.0%})")
    out.append(f"{'시나리오':<26}{'통과':>6}  실패 항목 (횟수)  [시뮬레이터 이탈]")
    for sid, rs in by_sc.items():
        n_pass = sum(r["passed"] for r in rs)
        fails = Counter(c["name"] for r in rs for c in r["checks"] if c["passed"] is False)
        drift = sum(1 for r in rs for c in r.get("diagnostics", [])
                    if c["name"] == "sim_faithful" and c["passed"] is False)
        fail_txt = ", ".join(f"{k}×{v}" for k, v in fails.most_common())
        out.append(f"{sid:<26}{n_pass:>3}/{len(rs):<3} {fail_txt}" + (f"  [이탈 {drift}]" if drift else ""))

    drifted = [(r["scenario_id"], r["repeat"], c["detail"]) for r in rows for c in r.get("diagnostics", [])
               if c["name"] == "sim_faithful" and c["passed"] is False]
    if drifted:
        out.append("\n시뮬레이터 이탈 (이 실행의 실패는 측정 잡음일 수 있음)")
        out += [f"  {sid} #{rep}: {what}" for sid, rep, what in drifted]

    violations = sum(len(r["procedure"]["violations"]) for r in rows)
    out.append(f"\n절차 위반 {violations}건")

    # -- LLM 호출 (출처별, 모니터링 집계와 같은 정의) --
    for d in run_dirs:
        llm = aggregate(d)["llm"]["by_source"]
        if not llm:
            continue
        out.append(f"\n{'LLM 출처':<10}{'호출':>6}{'입력 토큰':>12}{'출력 토큰':>11}{'지연 중앙값':>12}{'p90':>8}"
                   + (f"   ({d.name})" if len(run_dirs) > 1 else ""))
        for src, x in llm.items():
            out.append(f"{src:<10}{x['calls']:>6}{x['prompt_tokens']:>12,}{x['completion_tokens']:>11,}"
                       f"{x['ms']['p50']:>10,}ms{x['ms']['p90']:>6,}ms" + (f"  오류 {x['errors']}" if x["errors"] else ""))

    lat = sorted(rep["timings_ms"]["total"] for r in rows for rep in r["replies"])
    if lat:
        out.append(f"\n턴 응답 지연 중앙값 {st.median(lat):,.0f}ms  p90 {lat[int(len(lat) * .9)]:,.0f}ms")
    return "\n".join(out)


if __name__ == "__main__":
    print(summarize([Path(p) for p in sys.argv[1:]]))
