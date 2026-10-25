"""APO 후보 테스트 실행기 (scripts/apo_test.py) — 계획 검사·명세 생성·arm 환경·결과 표. 실제 재현 실행은 하지 않는다."""
import json

import pytest
import yaml

from csagent.agent.protocol import OVERRIDE_ENV
from scripts.apo_test import PlanError, _env, plan, table

SPEC = {"case": "runs/x/s3_b0_defect:0", "at": "t3", "mode": "turn", "repeat": 5, "fail_if": [{"repeat": True}],
        "customer": "recorded"}


def _exp(tmp_path):
    exp = tmp_path / "exp"
    exp.mkdir()
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))
    (exp / "source.json").write_text(json.dumps({"spec": str(spec)}))
    return exp, spec


def _plan(spec, **kw):
    return yaml.safe_dump({"target": {"spec": str(spec), "repeat": 5},
                           "variants": [{"name": "v1", "utterance": "사진은 어디에 올려요?"}],
                           "guards": [{"name": "g1", "spec": str(spec), "repeat": 2}],
                           "regression": {"scenarios": ["s3_b0_defect"], "repeat": 1}, **kw}, allow_unicode=True)


def test_plan_writes_specs_with_variant_utterance(tmp_path):
    exp, spec = _exp(tmp_path)
    assert plan(exp, _plan(spec)) == ["target", "v1", "g1"]
    v1 = yaml.safe_load((exp / "tests" / "v1.yaml").read_text(encoding="utf-8"))
    assert v1["utterance"] == "사진은 어디에 올려요?" and v1["fail_if"] == SPEC["fail_if"] and v1["repeat"] == 3
    assert "utterance" not in yaml.safe_load((exp / "tests" / "target.yaml").read_text(encoding="utf-8"))


def test_plan_target_must_be_triage_baseline(tmp_path):
    exp, spec = _exp(tmp_path)
    other = tmp_path / "other.json"
    other.write_text(json.dumps(SPEC))
    with pytest.raises(PlanError, match="triage의 재현 기준"):
        plan(exp, _plan(other))


def test_plan_rejects_duplicate_names(tmp_path):
    exp, spec = _exp(tmp_path)
    with pytest.raises(PlanError, match="겹친다"):
        plan(exp, _plan(spec, guards=[{"name": "v1", "spec": str(spec)}]))


def test_arm_env_sets_override_only_for_candidates(tmp_path, monkeypatch):
    exp, _ = _exp(tmp_path)
    monkeypatch.setenv(OVERRIDE_ENV, "/leftover")
    assert OVERRIDE_ENV not in _env(exp, "baseline")              # 남은 환경 변수로 baseline이 오염되지 않는다
    with pytest.raises(PlanError, match="후보 없음"):
        _env(exp, "a")
    (exp / "a").mkdir()
    (exp / "a" / "exchange_return.yaml").write_text("x")
    assert _env(exp, "a")[OVERRIDE_ENV] == str(exp / "a")


def test_table_uses_latest_row_per_cell(tmp_path):
    exp, _ = _exp(tmp_path)
    rows = [{"kind": "repro", "arm": "baseline", "test": "target", "fails": 5, "n": 5, "error": None},
            {"kind": "repro", "arm": "a", "test": "target", "fails": 4, "n": 5, "error": None},
            {"kind": "repro", "arm": "a", "test": "target", "fails": 1, "n": 5, "error": None},
            {"kind": "regression", "arm": "a", "test": "s3_b0_defect", "passed": 0, "n": 1, "failed": ["no_repeat"],
             "error": None}]
    (exp / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    md = table(exp)
    assert "| target | 실패 5/5 | 실패 1/5 |" in md and "통과 0/1 (no_repeat)" in md


def test_applied_arm_runs_repo_prompt_and_sorts_last(tmp_path, monkeypatch):
    exp, _ = _exp(tmp_path)
    monkeypatch.setenv(OVERRIDE_ENV, "/leftover")
    assert OVERRIDE_ENV not in _env(exp, "applied")
    rows = [{"kind": "repro", "arm": a, "test": "target", "fails": f, "n": 5, "error": None}
            for a, f in (("applied", 0), ("baseline", 5), ("b", 1))]
    (exp / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert table(exp).splitlines()[0] == "| 테스트 | baseline | b | applied |"


def test_run_all_gates_candidates_on_target(tmp_path, monkeypatch):
    import scripts.apo_test as t
    exp, spec = _exp(tmp_path)
    plan(exp, _plan(spec))
    for arm in ("good", "bad"):
        (exp / arm).mkdir()
        (exp / arm / "exchange_return.yaml").write_text("x")
    fails = {("baseline", "target"): 5, ("good", "target"): 0, ("bad", "target"): 5}
    ran = []

    def fake_run(exp_, arm, test):
        ran.append((arm, test))
        row = {"kind": "repro", "arm": arm, "test": test, "fails": fails.get((arm, test), 0), "n": 5, "error": None}
        t._record(exp_, row)
        return row

    def fake_regress(exp_, arm, sid):
        ran.append((arm, "reg:" + sid))
        row = {"kind": "regression", "arm": arm, "test": sid, "passed": 1, "n": 1, "failed": [], "error": None}
        t._record(exp_, row)
        return row

    monkeypatch.setattr(t, "run", fake_run)
    monkeypatch.setattr(t, "regress", fake_regress)
    t.run_all(exp)
    assert ("bad", "v1") not in ran and ("good", "v1") in ran and ("good", "g1") in ran      # target에서 못 고친 후보는 건너뜀
    assert ("good", "reg:s3_b0_defect") in ran and ("bad", "reg:s3_b0_defect") not in ran
    ran.clear()
    t.run_all(exp)
    assert ran == []                                                                           # 이어 돌리기: 다 있으면 아무것도 안 돈다
