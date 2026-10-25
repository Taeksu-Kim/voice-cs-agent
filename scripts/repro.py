"""재현기 실행 — 명세(yaml) 하나를 받아 N번 돌리고 실패 횟수를 낸다 (csagent/evaluation/repro.py, docs/10 5절).

    python scripts/repro.py spec.yaml
    python scripts/repro.py runs/20261021_010852_C/s3_b0_defect:0 --at t1 --mode turn --repeat 5 \\
        --fail len_over=150 --fail repeat=true

결과: runs/repro/<시각>_<키>/ — spec.json · repro.jsonl(반복별 실패 여부·맞은 술어·말·trace) · run_meta.json(버전) ·
llm_calls.jsonl. 수정 전후로 같은 명세를 두 번 돌려 실패 수를 비교한다 (run_meta의 commit·diff_sha로 어느 코드였는지 구분).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.agent.prompts import BASELINE_TASK
from csagent.agent.workers.specs import WORKERS
from csagent.config import ROOT
from csagent.evaluation.repro import ReproSpec, find_scenario, load_case, run_step, run_turn
from csagent.evaluation.scoring import LLMJudge
from csagent.evaluation.simulator import LLMCustomer
from csagent.factory import build_agent, build_llm
from csagent.llm_log import LLMCallLog
from csagent.obs import LLM_CALLS_FILE, log_context, setup_logging
from csagent.stack import record_versions, remote_stack


# LLM 없이 채점되는 항목 (scoring.rule_checks·path_checks)
RULE_CHECKS = {"disposition", "tools_required", "tools_forbidden", "milestones", "critical_exec", "no_repeat", "finished",
               "required_in_time", "procedure", "voice_length", "voice_format", "gold_doc_retrieved"}


def _value(v: str):
    return True if v == "true" else False if v == "false" else int(v) if v.isdigit() else v


def parse() -> ReproSpec:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="명세 yaml 또는 case (<폴더>/<시나리오:반복 | call_id>)")
    ap.add_argument("--at")
    ap.add_argument("--mode", default="turn")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--fail", action="append", default=[], help="술어=값 (여러 번)")
    ap.add_argument("--customer", default="recorded", help="continue: recorded(기록된 고객 말) | simulator")
    args = ap.parse_args()
    if args.spec.endswith((".yaml", ".yml")):
        return ReproSpec.from_dict(yaml.safe_load(Path(args.spec).read_text(encoding="utf-8")))
    fail_if = [{k: _value(v)} for k, _, v in (f.partition("=") for f in args.fail)]
    return ReproSpec.from_dict({"case": args.spec, "at": args.at, "mode": args.mode, "repeat": args.repeat,
                                "fail_if": fail_if, "customer": args.customer})


def main() -> None:
    spec = parse()
    source, row = load_case(spec.case)
    stack = remote_stack()
    key = Path(spec.case).name.replace(":", "_r")
    out = ROOT / "runs" / "repro" / f"{datetime.now():%Y%m%d_%H%M%S}_{key}_{spec.at}_{spec.mode}"
    out.mkdir(parents=True)
    setup_logging(out)
    log = LLMCallLog(out / LLM_CALLS_FILE)
    (out / "spec.json").write_text(json.dumps(vars(spec), ensure_ascii=False, indent=1), encoding="utf-8")
    meta = record_versions(stack, out)
    (out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"재현 {spec.case} @{spec.at} {spec.mode} ×{spec.repeat}  (코드 {meta['commit']}"
          f"{' +' + meta['diff_sha'] if meta.get('diff_sha') else ''})  → {out.relative_to(ROOT)}", flush=True)

    if spec.mode == "step":
        results = run_step(build_llm(log=log, source="repro"), source, row, spec)
    else:
        sc = find_scenario(row)
        agent = build_agent("C", log=log, tools=stack.tools)
        customer_llm = build_llm(log=log, source="customer")
        judge = None
        # 판정기는 술어에 LLM 판정 항목(grounded 등)이 있을 때만 — 규칙 채점(milestones 등)만 보면 판정 호출이 시간 대부분을 쓴다
        needs_judge = any(k == "check" and v not in RULE_CHECKS for p in spec.fail_if for k, v in p.items())
        if spec.mode == "continue" and needs_judge:
            instructions = {"baseline": BASELINE_TASK, **{n: w.policy.task for n, w in WORKERS.items()}}
            judge = LLMJudge(build_llm(log=log, source="judge"), instructions)
        results = []
        for i in range(spec.repeat):
            with log_context(scenario=sc.id, repeat=i):
                results.append(run_turn(agent, stack, sc, row, spec,
                                        customer=LLMCustomer(customer_llm), judge=judge))
            r = results[-1]
            print(f"  [{'FAIL' if r['failed'] else 'ok  '}] {i}  {'; '.join(r['hits'])}", flush=True)
            for t in r["texts"]:
                print(f"         {len(t):3d}자 {t[:110]}", flush=True)

    with (out / "repro.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(results):
            f.write(json.dumps({"i": i, **r}, ensure_ascii=False, default=str) + "\n")
    if spec.mode == "step":
        for i, r in enumerate(results):
            print(f"  [{'FAIL' if r['failed'] else 'ok  '}] {i}  {len(r['texts'][0]):3d}자 {r['texts'][0][:110]}")
    n = sum(r["failed"] for r in results)
    print(f"\n실패 {n}/{len(results)}  ({', '.join(f'{k}: {v}' for p in spec.fail_if for k, v in p.items())})")


if __name__ == "__main__":
    main()
