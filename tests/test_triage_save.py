"""triage 보고서 저장 (scripts/triage_save.py) — 형식 검사와 저장 위치."""
import json

import pytest

from scripts.triage_save import ReportError, parse, save

REPORT = """---
case: runs/x/s3_b0_defect:0
issues:
  - signal: repeat @t3
    summary: 같은 반박에 같은 답
    layer: prompt
    secondary: ""
    confidence: reproduced
    baseline: "5/5"
    spec: runs/repro/a/spec.json
    proposal: 반복 금지 지시를 구체화
    next: apo
  - signal: over_len @t1
    summary: 세 스텝의 말이 한 턴에 이어 붙음
    layer: code
    confidence: hypothesis
    proposal: 링크 발송 뒤 안내를 다음 턴으로
    next: human
---

## 문제 1 — repeat @t3
- t3 spoken 인용
"""


def test_parse_reads_issues_and_body():
    head, body = parse(REPORT)
    assert [i["layer"] for i in head["issues"]] == ["prompt", "code"] and body.startswith("## 문제 1")


@pytest.mark.parametrize("change, msg", [
    (("layer: prompt", "layer: vibes"), r"issues\[0\] layer는"),
    (("next: human", "next: fix_it"), r"issues\[1\] next는"),
    (('baseline: "5/5"', 'baseline: ""'), "baseline"),                 # 재현됨인데 기준치가 없다
    (("summary: 같은 반박에 같은 답", 'summary: ""'), "빠진 필드: summary"),
    (('secondary: ""', "secondary: prompt"), "secondary가 layer와 같다"),
    (("issues:", "items:"), "issues는"),
])
def test_parse_rejects_bad_reports(change, msg):
    with pytest.raises(ReportError, match=msg):
        parse(REPORT.replace(*change, 1))


def test_save_writes_case_report_and_one_index_row_per_issue(tmp_path):
    case = tmp_path / "case1"
    case.mkdir()
    index = tmp_path / "triage" / "index.jsonl"
    with pytest.raises(ReportError, match="case.json"):
        save(case, REPORT, index)
    (case / "case.json").write_text("{}")
    save(case, REPORT, index)
    assert (case / "triage.md").read_text(encoding="utf-8").startswith("---")
    rows = [json.loads(l) for l in index.read_text(encoding="utf-8").splitlines()]
    assert [(r["issue"], r["layer"], r["next"]) for r in rows] == [(0, "prompt", "apo"), (1, "code", "human")]
    assert rows[1]["case"] == "runs/x/s3_b0_defect:0" and rows[1]["baseline"] == ""
