"""triage 보고서 저장 — 형식을 검사하고 케이스 폴더의 triage.md와 누적 목록 runs/triage/index.jsonl에 남긴다.

    python scripts/triage_save.py runs/cases/<시각>/<케이스> < report.md

triage 스킬(.claude/skills/triage)은 파일을 직접 쓰지 않고 이 스크립트로만 저장한다 — 저장 위치와 형식이 고정된다.
형식: reference/report_template.md (맨 위 YAML 블록 — case + issues 목록). 한 케이스에 원인이 다른 문제가 여럿이면
issues에 하나씩 둔다 (보조 원인 secondary는 같은 문제의 다른 원인일 때만). 목록에는 문제마다 한 줄.
같은 케이스를 다시 분석하면 triage.md는 덮어쓰고 목록에는 더 쌓인다.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.config import ROOT

LAYERS = ("prompt", "guide", "doc", "code", "judge", "env")
CONFIDENCE = ("reproduced", "hypothesis")
NEXT = ("apo", "human", "judge_set", "none")
REQUIRED = ("signal", "summary", "layer", "confidence", "proposal", "next")
FIELDS = ("signal", "summary", "layer", "secondary", "confidence", "baseline", "spec", "proposal", "next")


class ReportError(ValueError):
    pass


def _check_issue(n: int, issue: dict) -> None:
    where = f"issues[{n}]"
    missing = [k for k in REQUIRED if not issue.get(k)]
    if missing:
        raise ReportError(f"{where} 빠진 필드: {', '.join(missing)}")
    for key, allowed in (("layer", LAYERS), ("confidence", CONFIDENCE), ("next", NEXT)):
        if issue[key] not in allowed:
            raise ReportError(f"{where} {key}는 {' | '.join(allowed)}: {issue[key]}")
    if issue.get("secondary") and issue["secondary"] not in LAYERS:
        raise ReportError(f"{where} secondary는 {' | '.join(LAYERS)} 또는 빈 값: {issue['secondary']}")
    if issue.get("secondary") == issue["layer"]:
        raise ReportError(f"{where} secondary가 layer와 같다")
    if issue["confidence"] == "reproduced" and not (issue.get("baseline") and issue.get("spec")):
        raise ReportError(f"{where} 재현됨이면 baseline(수정 전 실패 수)과 spec(재현 명세)이 있어야 한다")


def parse(text: str) -> tuple[dict, str]:
    """(머리 블록, 본문). 머리 블록 = 첫 줄 --- 부터 다음 --- 까지의 YAML: {case, issues: [...]}."""
    lines = text.strip().splitlines()
    if not lines or lines[0].strip() != "---":
        raise ReportError("보고서는 --- 로 시작하는 머리 블록이 있어야 한다")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise ReportError("머리 블록이 --- 로 닫히지 않았다") from None
    try:
        head = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError as e:
        raise ReportError(f"머리 블록 YAML 오류: {e}") from None
    if not head.get("case"):
        raise ReportError("빠진 필드: case")
    issues = head.get("issues")
    if not isinstance(issues, list) or not issues:
        raise ReportError("issues는 문제 하나 이상의 목록이어야 한다")
    for n, issue in enumerate(issues):
        if not isinstance(issue, dict):
            raise ReportError(f"issues[{n}]는 필드 목록이어야 한다")
        _check_issue(n, issue)
    body = "\n".join(lines[end + 1:]).strip()
    if not body:
        raise ReportError("본문(근거)이 비었다")
    return head, body


def save(case_dir: Path, text: str, index: Path) -> list[dict]:
    if not (case_dir / "case.json").exists():
        raise ReportError(f"케이스 폴더가 아니다 (case.json 없음): {case_dir}")
    head, _ = parse(text)
    (case_dir / "triage.md").write_text(text.strip() + "\n", encoding="utf-8")
    at = datetime.now().isoformat(timespec="seconds")
    rows = [{"at": at, "case_dir": str(case_dir), "case": head["case"], "issue": n,
             **{k: issue.get(k, "") for k in FIELDS}} for n, issue in enumerate(head["issues"])]
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    case_dir = Path(sys.argv[1])
    case_dir = case_dir if case_dir.is_absolute() else ROOT / case_dir
    try:
        rows = save(case_dir, sys.stdin.read(), ROOT / "runs" / "triage" / "index.jsonl")
    except ReportError as e:
        sys.exit(f"저장 안 함: {e}")
    print(f"저장: {case_dir / 'triage.md'} — 문제 {len(rows)}개")
    for r in rows:
        print(f"  [{r['layer']} · {r['confidence']} · 다음 {r['next']}] {r['signal']}: {r['summary']}")


if __name__ == "__main__":
    main()
