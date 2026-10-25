"""APO 후보 테스트 — 지금 지시문(baseline)과 후보마다 같은 테스트를 돌려 실패 수를 비교한다. 저장소 지시문은 건드리지 않는다.

    python scripts/apo_test.py plan <실험 폴더> < tests.yaml        # 테스트 계획 검사·저장 (명세 파일 생성)
    python scripts/apo_test.py run <실험 폴더> --arm <baseline|후보> --test <이름>
    python scripts/apo_test.py regress <실험 폴더> --arm <baseline|후보> --scenario <시나리오 id>
    python scripts/apo_test.py table <실험 폴더>                    # 결과 표 → results.md
    python scripts/apo_test.py verify <실험 폴더> --candidate <후보>   # 사람이 반영한 뒤: 저장소 지시문으로 같은 테스트 전부 → applied 열
    python scripts/apo_test.py run-all <실험 폴더>                  # 계획의 모든 칸을 순서대로 (Claude 밖에서 — 단계별 거르기)

tests.yaml:
    target:                       # triage의 재현 기준 — 바꾸지 않는다
      spec: runs/repro/<…>/spec.json
      repeat: 5
    variants:                     # 일반화 — 같은 상황, 다른 고객 말 (target 명세에 utterance만 바꿔 넣는다)
      - {name: v1, utterance: "…", why: "…"}
    guards:                       # 퇴행 확인 — 다른 재현 명세 (예: 같은 통화의 다른 문제)
      - {name: g1, spec: runs/repro/<…>/spec.json, repeat: 3}
    suite: exchange_return        # 케이스 모음의 그 업무 항목 전부 → suite_<id> 테스트 (scripts/suite.py)
    regression:                   # 관련 시나리오 평가 (run_eval.py --only) — 기본 규칙 채점만 (judge: true면 LLM 판정까지)
      scenarios: [s3_b0_defect, …]
      repeat: 1

후보 arm은 CSAGENT_PROTOCOL_OVERRIDE=<실험 폴더>/<후보>로 돈다 (csagent/agent/protocol.py). 결과는 <실험 폴더>/results.jsonl에 쌓인다.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.agent.protocol import OVERRIDE_ENV
from csagent.config import ROOT

PY = sys.executable
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
_FAILS = re.compile(r"실패 (\d+)/(\d+)")


class PlanError(ValueError):
    pass


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _load_spec(path: str) -> dict:
    p = _abs(path)
    if not p.exists():
        raise PlanError(f"명세 없음: {path}")
    d = json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else yaml.safe_load(p.read_text(encoding="utf-8"))
    if not d.get("fail_if"):
        raise PlanError(f"fail_if 없는 명세: {path}")
    return d


def plan(exp: Path, text: str) -> list[str]:
    """테스트 계획 검사 → tests/<이름>.yaml 명세 파일. 테스트 이름 목록을 돌려준다."""
    if not (exp / "source.json").exists():
        raise PlanError(f"실험 폴더가 아니다 (source.json 없음): {exp}")
    p = yaml.safe_load(text) or {}
    target = p.get("target") or {}
    base = _load_spec(target.get("spec", ""))
    source = json.loads((exp / "source.json").read_text(encoding="utf-8"))
    if source.get("spec") and _abs(source["spec"]) != _abs(target["spec"]):
        raise PlanError(f"target은 triage의 재현 기준이어야 한다: {source['spec']}")
    specs = {"target": {**base, "repeat": int(target.get("repeat", 5))}}
    for v in p.get("variants") or []:
        if not _NAME.match(str(v.get("name", ""))) or not v.get("utterance"):
            raise PlanError(f"변형은 name(영문 소문자)과 utterance가 있어야 한다: {v}")
        if base.get("mode") == "step":
            raise PlanError("target이 step 모드라 고객 말 변형을 쓸 수 없다")
        specs[v["name"]] = {**base, "utterance": v["utterance"], "repeat": int(v.get("repeat", 3))}
    for g in p.get("guards") or []:
        if not _NAME.match(str(g.get("name", ""))):
            raise PlanError(f"guard 이름: {g}")
        specs[g["name"]] = {**_load_spec(g.get("spec", "")), "repeat": int(g.get("repeat", 3))}
    if p.get("suite"):   # 케이스 모음 (scripts/suite.py) — 그 업무의 항목 전부를 퇴행 확인으로
        from scripts.suite import load as load_suite
        entries = load_suite(ROOT, str(p["suite"]))
        if not entries:
            raise PlanError(f"케이스 모음이 비었다: {p['suite']}")
        for e in entries:
            specs[f"suite_{e['id']}"] = {**e["spec"], "repeat": 3 if e["kind"] == "defect" else 2}
    reg = p.get("regression") or {}
    if reg and not all(re.match(r"^[a-z0-9_]+$", s) for s in reg.get("scenarios") or []):
        raise PlanError("regression.scenarios는 시나리오 id 목록")
    n_suite = sum(k.startswith("suite_") for k in specs)
    if len(specs) != 1 + len(p.get("variants") or []) + len(p.get("guards") or []) + n_suite:
        raise PlanError("테스트 이름이 겹친다")
    (exp / "tests").mkdir(exist_ok=True)
    for name, spec in specs.items():
        spec = {k: v for k, v in spec.items() if v is not None}
        (exp / "tests" / f"{name}.yaml").write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False),
                                                     encoding="utf-8")
    (exp / "tests.yaml").write_text(text.strip() + "\n", encoding="utf-8")
    return list(specs)


def _env(exp: Path, arm: str) -> dict:
    """baseline·applied는 저장소 지시문 그대로 (applied = 사람이 후보를 반영한 뒤), 후보는 그 폴더를 덮어써서."""
    env = {k: v for k, v in os.environ.items() if k != OVERRIDE_ENV}
    if arm not in ("baseline", "applied"):
        d = exp / arm
        if not _NAME.match(arm) or not any(d.glob("*.yaml")):
            raise PlanError(f"후보 없음: {arm}")
        env[OVERRIDE_ENV] = str(d)
    return env


@contextlib.contextmanager
def _exclusive():
    """재현·평가는 한 번에 하나 — 로컬 벡터 DB(Milvus Lite) 파일은 한 프로세스만 연다. 동시에 부르면 앞 것이 끝날 때까지 기다린다."""
    lock = ROOT / "runs" / ".apo_test.lock"
    lock.parent.mkdir(exist_ok=True)
    with lock.open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def done(exp: Path, kind: str, arm: str, test: str) -> bool:
    """그 칸에 오류 아닌 결과가 이미 있나 (이어 돌릴 때 건너뛴다)."""
    if not (exp / "results.jsonl").exists():
        return False
    rows = [json.loads(l) for l in (exp / "results.jsonl").open(encoding="utf-8")]
    last = [r for r in rows if (r["kind"], r["arm"], r["test"]) == (kind, arm, test)]
    return bool(last) and not last[-1].get("error")


def _record(exp: Path, row: dict) -> None:
    with (exp / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(exp: Path, arm: str, test: str) -> dict:
    spec = exp / "tests" / f"{test}.yaml"
    if not spec.exists():
        raise PlanError(f"테스트 없음: {test} (plan 먼저)")
    with _exclusive():
        p = subprocess.run([PY, "scripts/repro.py", str(spec)], cwd=ROOT, env=_env(exp, arm),
                           capture_output=True, text=True)
    m = _FAILS.search(p.stdout)
    out = re.search(r"→ (runs/repro/\S+)", p.stdout)
    texts = re.findall(r"^\s+\d+자 (.*)$", p.stdout, re.M)
    row = {"kind": "repro", "arm": arm, "test": test, "fails": int(m[1]) if m else None, "n": int(m[2]) if m else None,
           "out": out[1] if out else None, "texts": texts[:3], "error": None if m else (p.stderr or p.stdout)[-500:]}
    _record(exp, row)
    return row


def regress(exp: Path, arm: str, scenario: str) -> dict:
    reg = (yaml.safe_load((exp / "tests.yaml").read_text(encoding="utf-8")) or {}).get("regression") or {}
    if scenario not in (reg.get("scenarios") or []):
        raise PlanError(f"계획에 없는 회귀 시나리오: {scenario}")
    with _exclusive():
        # 기본은 규칙 채점만 — 말 내용(근거·약속)은 review에서 출력을 직접 본다. regression.judge: true면 LLM 판정까지
        cmd = [PY, "scripts/run_eval.py", "--agent", "C", "--only", scenario, "--repeat", str(reg.get("repeat", 1))]
        p = subprocess.run(cmd + ([] if reg.get("judge") else ["--no-judge"]), cwd=ROOT, env=_env(exp, arm),
                           capture_output=True, text=True)
    out = re.search(r"결과 폴더 (runs/\S+)", p.stdout)
    rows = []
    if out and (ROOT / out[1] / "results.jsonl").exists():
        rows = [json.loads(l) for l in (ROOT / out[1] / "results.jsonl").open(encoding="utf-8")
                if json.loads(l)["scenario_id"] == scenario]
    row = {"kind": "regression", "arm": arm, "test": scenario, "passed": sum(r["passed"] for r in rows), "n": len(rows),
           "failed": sorted({c["name"] for r in rows for c in r["checks"] if c["passed"] is False}),
           "out": out[1] if out else None, "error": None if rows else (p.stderr or p.stdout)[-500:]}
    _record(exp, row)
    return row


def verify(exp: Path, candidate: str, regression: bool = True) -> dict:
    """반영 확인: 저장소 지시문이 후보와 같은가 → 같은 테스트를 applied arm으로 전부 다시. 결과는 applied.json과 results.jsonl."""
    from csagent.agent.protocol import PROTOCOL_DIR
    files = sorted((exp / candidate).glob("*.yaml"))
    if not files:
        raise PlanError(f"후보 없음: {candidate}")
    if not (exp / "tests").exists():
        raise PlanError("테스트 계획이 없다 (plan 먼저)")
    match = {f.name: (PROTOCOL_DIR / f.name).read_text(encoding="utf-8") == f.read_text(encoding="utf-8") for f in files}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    info = {"candidate": candidate, "files_match": match, "commit": commit}
    (exp / "applied.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    if not all(match.values()):
        print(f"!! 저장소 지시문이 후보와 다르다: {match} — 반영한 내용 그대로 확인을 계속한다", flush=True)
    for spec in sorted((exp / "tests").glob("*.yaml")):
        r = run(exp, "applied", spec.stem)
        print(f"  applied {spec.stem}: 실패 {r['fails']}/{r['n']}" if not r["error"] else f"  applied {spec.stem}: 오류", flush=True)
    if regression:
        reg = (yaml.safe_load((exp / "tests.yaml").read_text(encoding="utf-8")) or {}).get("regression") or {}
        for sid in reg.get("scenarios") or []:
            r = regress(exp, "applied", sid)
            print(f"  applied 회귀 {sid}: 통과 {r['passed']}/{r['n']} {r['failed'] or ''}", flush=True)
    return info


def _cell(exp: Path, kind: str, arm: str, test: str) -> dict | None:
    if not (exp / "results.jsonl").exists():
        return None
    rows = [json.loads(l) for l in (exp / "results.jsonl").open(encoding="utf-8")]
    last = [r for r in rows if (r["kind"], r["arm"], r["test"]) == (kind, arm, test)]
    return last[-1] if last else None


def run_all(exp: Path, regression: bool = True, only: list[str] | None = None) -> None:
    """모든 칸을 순서대로. 단계별 거르기: target에서 baseline보다 실패가 줄지 않은 후보는 이후 테스트를 건너뛴다.
    회귀는 baseline과 target을 통과한(실패 0) 후보만. 이미 결과가 있는 칸은 건너뛴다 (이어 돌리기)."""
    if not (exp / "tests").exists():
        raise PlanError("테스트 계획이 없다 (plan 먼저)")
    arms = ["baseline"] + sorted(d.name for d in exp.iterdir() if d.is_dir() and any(d.glob("*.yaml"))
                                 and d.name != "tests" and (not only or d.name in only))
    tests = ["target"] + sorted(p.stem for p in (exp / "tests").glob("*.yaml") if p.stem != "target")

    def go(arm: str, test: str) -> dict:
        r = _cell(exp, "repro", arm, test)
        if r is None or r.get("error"):
            r = run(exp, arm, test)
        print(f"  {arm:28s} {test:16s} " + (f"실패 {r['fails']}/{r['n']}" if not r.get("error") else "오류"), flush=True)
        return r

    base = go("baseline", "target")
    alive = ["baseline"]
    for arm in arms[1:]:
        r = go(arm, "target")
        if not r.get("error") and base.get("fails") is not None and r["fails"] < base["fails"]:
            alive.append(arm)
        else:
            print(f"  → {arm}: target에서 나아지지 않아 이후 테스트 건너뜀", flush=True)
    for test in tests[1:]:
        for arm in alive:
            go(arm, test)
    if regression:
        reg = (yaml.safe_load((exp / "tests.yaml").read_text(encoding="utf-8")) or {}).get("regression") or {}
        reg_arms = ["baseline"] + [a for a in alive[1:] if (_cell(exp, "repro", a, "target") or {}).get("fails") == 0]
        for sid in reg.get("scenarios") or []:
            for arm in reg_arms:
                r = _cell(exp, "regression", arm, sid)
                if r is None or r.get("error"):
                    r = regress(exp, arm, sid)
                print(f"  {arm:28s} 회귀 {sid:22s} 통과 {r['passed']}/{r['n']} {r['failed'] or ''}", flush=True)


def table(exp: Path) -> str:
    rows = [json.loads(l) for l in (exp / "results.jsonl").open(encoding="utf-8")] if (exp / "results.jsonl").exists() else []
    latest = {(r["kind"], r["arm"], r["test"]): r for r in rows}       # 같은 칸을 다시 돌렸으면 마지막 것
    arms = ["baseline"] + sorted({a for _, a, _ in latest if a not in ("baseline", "applied")})
    arms += ["applied"] if any(a == "applied" for _, a, _ in latest) else []
    tests = list(dict.fromkeys(t for k, _, t in latest if k == "repro"))
    regs = list(dict.fromkeys(t for k, _, t in latest if k == "regression"))
    cell = lambda r: "-" if r is None else ("오류" if r.get("error") else
                                           f"실패 {r['fails']}/{r['n']}" if r["kind"] == "repro" else
                                           f"통과 {r['passed']}/{r['n']}" + (f" ({', '.join(r['failed'])})" if r["failed"] else ""))
    lines = ["| 테스트 | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for t in tests:
        lines.append(f"| {t} | " + " | ".join(cell(latest.get(("repro", a, t))) for a in arms) + " |")
    for t in regs:
        lines.append(f"| 회귀 {t} | " + " | ".join(cell(latest.get(("regression", a, t))) for a in arms) + " |")
    md = "\n".join(lines)
    (exp / "results.md").write_text(md + "\n", encoding="utf-8")
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "run", "regress", "table", "verify", "run-all"))
    ap.add_argument("exp")
    ap.add_argument("--arm")
    ap.add_argument("--test")
    ap.add_argument("--scenario")
    ap.add_argument("--candidate")
    ap.add_argument("--no-regress", action="store_true")
    ap.add_argument("--force", action="store_true", help="결과가 있는 칸도 다시")
    ap.add_argument("--arms", help="run-all: 이 후보만 (쉼표로) — 유력한 후보 하나만 돌릴 때")
    a = ap.parse_args()
    exp = _abs(a.exp)
    try:
        if a.cmd == "plan":
            print("테스트:", ", ".join(plan(exp, sys.stdin.read())))
        elif a.cmd in ("run", "regress") and not a.force and done(
                exp, "repro" if a.cmd == "run" else "regression", a.arm or "baseline", a.test or a.scenario):
            print(f"이미 결과 있음 — 건너뜀: {a.arm or 'baseline'} {a.test or a.scenario} (다시 돌리려면 --force)")
        elif a.cmd == "run":
            r = run(exp, a.arm or "baseline", a.test)
            print(json.dumps({k: r[k] for k in ("arm", "test", "fails", "n", "out", "error")}, ensure_ascii=False))
            for t in r["texts"]:
                print("  ", t)
        elif a.cmd == "regress":
            r = regress(exp, a.arm or "baseline", a.scenario)
            print(json.dumps({k: r[k] for k in ("arm", "test", "passed", "n", "failed", "out", "error")}, ensure_ascii=False))
        elif a.cmd == "run-all":
            run_all(exp, regression=not a.no_regress, only=a.arms.split(",") if a.arms else None)
            print(table(exp))
        elif a.cmd == "verify":
            if not a.candidate:
                raise PlanError("--candidate 필요")
            verify(exp, a.candidate, regression=not a.no_regress)
            print(table(exp))
        else:
            print(table(exp))
    except PlanError as e:
        sys.exit(f"중단: {e}")


if __name__ == "__main__":
    main()
