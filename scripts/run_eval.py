"""평가 실행: 시나리오 전부를 고객 시뮬레이터 ↔ 에이전트로 돌리고 채점한다.

    python scripts/run_eval.py [--only s1_] [--no-judge] [--repeat 1]

결과 (runs/<시각>/): results.jsonl (통화 기록 + 채점), events.jsonl (운영 이벤트),
llm_calls.jsonl (LLM 입출력), errors.jsonl (오류로 건너뛴 시나리오). 화면에는 시나리오별 결과와 요약 표.
실행 중 지표 보기: python scripts/monitor.py runs/<시각> --follow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.config import ROOT
from csagent.agent.prompts import BASELINE_TASK
from csagent.agent.workers.specs import WORKERS
from csagent.factory import build_agent, build_llm
from csagent.stack import remote_stack
from csagent.llm_log import LLMCallLog
from csagent.obs import LLM_CALLS_FILE, event, log_context, setup_logging
from csagent.evaluation.runner import run_scenario
from csagent.evaluation.scenario import load_scenarios
from csagent.evaluation.scoring import LLMJudge, score
from csagent.evaluation.simulator import LLMCustomer
from scripts.summarize_run import ERRORS_FILE, summarize

MAX_ERROR_STREAK = 3   # 연속 오류가 이만큼이면 실행 중단


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=[""], help="시나리오 id 접두어 (여러 개면 하나라도 맞으면)")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--agent", default="C", help="A0 | A1 | C")
    ap.add_argument("--dir", nargs="+", default=[str(ROOT / "scenarios")],
                    help="시나리오 폴더 (보류 세트 등). 여러 개면 한 실행에서 모두 돈다")
    args = ap.parse_args()

    # 기동 헬스체크 (docs/08 8절) — 도구 없이 돌면 결과가 전부 오염되므로 결과 폴더를 만들기 전에 끝낸다
    stack = remote_stack()
    agent_kind = args.agent.upper().rstrip("'")
    out_dir = ROOT / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{agent_kind}"
    out_dir.mkdir(parents=True)
    setup_logging(out_dir)
    log = LLMCallLog(out_dir / LLM_CALLS_FILE)   # 모든 LLM 호출의 입출력 (trace의 call_id로 찾는다)
    print(f"결과 폴더 {out_dir.relative_to(ROOT)}", flush=True)
    # 버전 기록 (docs/08 9절) — 결과가 달라졌을 때 카탈로그 변경 때문인지 구분
    meta = {"args": vars(args), **stack.versions()}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if stack.tools.warnings:
        print(f"!! 도구 경고: {stack.tools.warnings}", flush=True)

    tools = stack.tools
    agent = build_agent(args.agent, log=log, tools=tools)
    customer_llm = build_llm(log=log, source="customer")
    instructions = {"baseline": BASELINE_TASK, **{n: w.policy.task for n, w in WORKERS.items()}}
    judge = None if args.no_judge else LLMJudge(build_llm(log=log, source="judge"), instructions)
    scenarios = [x for d in args.dir for x in load_scenarios(Path(d)) if x.id.startswith(tuple(args.only))]
    rows, errors, streak = [], 0, 0
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f, \
         (out_dir / ERRORS_FILE).open("w", encoding="utf-8") as ef:
        for sc in scenarios:
            for rep in range(args.repeat):
                with log_context(scenario=sc.id, repeat=rep):
                    try:
                        tr = run_scenario(agent, LLMCustomer(customer_llm), sc, stack)
                        sco = score(sc, tr, tools, judge)
                    except Exception as e:
                        # LLM 재시도(csagent/llm.py)까지 실패한 경우 — 이 시나리오만 건너뛴다.
                        # 연달아 실패하면 서버가 멈춘 것이므로 실행을 끝낸다 (나머지를 다 실패로 채우지 않는다)
                        errors, streak = errors + 1, streak + 1
                        ef.write(json.dumps({"scenario_id": sc.id, "repeat": rep, "error": repr(e)},
                                            ensure_ascii=False) + "\n")
                        ef.flush()
                        event(logging.getLogger("csagent.eval"), "scenario_error", logging.ERROR, error=repr(e))
                        print(f"[ERR ] {sc.id}  {e!r}", flush=True)
                        if streak >= MAX_ERROR_STREAK:
                            print(f"!! 연속 {streak}건 실패 — 실행 중단 (LLM 서버 확인)", flush=True)
                            break
                        continue
                streak = 0
                rows.append((sc, tr, sco))
                f.write(json.dumps({"repeat": rep, **tr.to_dict(),
                                    "passed": sco.passed,
                                    "checks": [c.__dict__ for c in sco.checks],
                                    "diagnostics": [c.__dict__ for c in sco.diagnostics],
                                    "procedure": {"executions": sco.procedure.executions,
                                                  "violations": sco.procedure.violations,
                                                  "gate_blocks": sco.procedure.gate_blocks}},
                                   ensure_ascii=False, default=str) + "\n")
                f.flush()   # 실행 중에도 결과를 읽을 수 있게
                with log_context(scenario=sc.id, repeat=rep):
                    event(logging.getLogger("csagent.eval"), "scenario_result", call_id=tr.state.call_id,
                          passed=sco.passed, failed=[c.name for c in sco.failed()],
                          disposition=tr.disposition, end_reason=tr.end_reason, turns=len(tr.replies),
                          sim_drift=any(c.passed is False for c in sco.diagnostics))
                _print_row(sc.id, tr, sco)
            if streak >= MAX_ERROR_STREAK:
                break

    blocks = sum(len(sco.procedure.gate_blocks) for _, _, sco in rows)
    print(f"\n[{agent.name}] 게이트 차단 {blocks}건 | 오류로 건너뜀 {errors}건 | 결과 {out_dir.relative_to(ROOT)}\n")
    print(summarize([out_dir]), flush=True)


def _print_row(sid: str, tr, sco) -> None:
    mark = "PASS" if sco.passed else "FAIL"
    print(f"[{mark}] {sid}  턴 {len(tr.replies)}  결과 {tr.disposition}  ({tr.end_reason})", flush=True)
    for c in sco.failed():
        print(f"    ✗ {c.name}: {c.detail}", flush=True)


if __name__ == "__main__":
    main()
