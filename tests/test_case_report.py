"""케이스 리포트 (scripts/case_report.py) — 산출물이 있는 만큼만 묶는다."""
import json

from scripts.case_report import build
from tests.test_triage_save import REPORT

CASE = {"case": "runs/x/s3_b0_defect:0", "scenario": "s3_b0_defect", "disposition": "link_sent", "end_reason": "agent_ended",
        "signals": [{"kind": "repeat", "level": "warn", "at": "t3", "detail": "같은 말"}], "guards": {}, "version": None}


def _case(tmp_path):
    d = tmp_path / "zz_report_test_case"
    d.mkdir()
    (d / "case.json").write_text(json.dumps(CASE, ensure_ascii=False), encoding="utf-8")
    return d


def test_before_triage_shows_signals_only(tmp_path):
    md = build(_case(tmp_path))
    assert "triage 전" in md and "| warn | repeat | `t3` |" in md and "## 3." not in md


def test_with_triage_lists_issues_and_todo(tmp_path):
    d = _case(tmp_path)
    (d / "triage.md").write_text(REPORT, encoding="utf-8")
    md = build(d)
    assert "| 1 | repeat @t3 | prompt | 재현됨 5/5 |" in md and "APO 실험 전" in md
    assert "## 3. 지시문 개선 (APO)" in md and "- 문제 2 (over_len @t1): 사람 검토" in md
