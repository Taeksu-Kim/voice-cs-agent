"""케이스 모음 (scripts/suite.py)과 apo-test 계획 연결."""
import json

import pytest
import yaml

from scripts.apo_test import plan
from scripts.suite import SuiteError, add_triage, add_variant, load, promote
from tests.test_apo_test import SPEC, _exp, _plan
from tests.test_triage_save import REPORT


def _triage_case(tmp_path):
    spec = tmp_path / "runs" / "repro" / "a" / "spec.json"
    spec.parent.mkdir(parents=True)
    spec.write_text(json.dumps(SPEC))
    case = tmp_path / "case1"
    case.mkdir()
    (case / "triage.md").write_text(REPORT.replace("runs/repro/a/spec.json", str(spec)), encoding="utf-8")
    return case


def test_triage_issue_becomes_defect_then_guard(tmp_path):
    case = _triage_case(tmp_path)
    item = add_triage(tmp_path, case, 0, "exchange_return")
    assert item["kind"] == "defect" and item["spec"]["fail_if"] == [{"repeat": True}] and "repeat" not in item["spec"]
    with pytest.raises(SuiteError, match="이미 있는 id|같은 명세"):
        add_triage(tmp_path, case, 0, "exchange_return")
    with pytest.raises(SuiteError, match="재현된 문제만"):
        add_triage(tmp_path, case, 1, "exchange_return")             # issue 1은 가설
    assert promote(tmp_path, "exchange_return", item["id"])["kind"] == "guard"


def test_variant_becomes_entry_and_plan_expands_suite(tmp_path, monkeypatch):
    exp, spec = _exp(tmp_path)
    plan(exp, _plan(spec))
    add_variant(tmp_path, exp, "v1", "exchange_return", "guard", item_id="photo_where")
    assert [x["id"] for x in load(tmp_path, "exchange_return")] == ["photo_where"]
    import scripts.apo_test as t
    monkeypatch.setattr(t, "ROOT", tmp_path)
    names = plan(exp, _plan(spec, suite="exchange_return"))
    assert "suite_photo_where" in names
    s = yaml.safe_load((exp / "tests" / "suite_photo_where.yaml").read_text(encoding="utf-8"))
    assert s["utterance"] == "사진은 어디에 올려요?" and s["repeat"] == 2
