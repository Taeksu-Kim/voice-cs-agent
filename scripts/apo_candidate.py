"""APO 후보 관리 — 지시문 수정 후보를 저장소 밖(runs/apo/)에 사본으로 만들고 검사한다. 저장소 지시문은 건드리지 않는다.

    python scripts/apo_candidate.py new <triage 케이스 폴더> <issue 번호>         # 실험 폴더 만들기 → 경로 출력
    python scripts/apo_candidate.py add <실험 폴더> <후보 이름> <프로토콜 이름> < 후보.yaml
    python scripts/apo_candidate.py note <실험 폴더> <후보 이름> < 근거.md      # 원칙 점검·모순 검토
    python scripts/apo_candidate.py proposal <실험 폴더> < 제안.md              # 최종 제안 (사람이 읽고 반영을 결정)

실험 폴더 runs/apo/<시각>_<케이스>_i<번호>/: source.json(triage 문제 그대로) · <후보>/<프로토콜>.yaml(파일 전체 사본) ·
<후보>/diff.patch(원본 대비) · <후보>/rationale.md · proposal.md.
후보로 돌리기: CSAGENT_PROTOCOL_OVERRIDE=<실험 폴더>/<후보> python scripts/repro.py … (csagent/agent/protocol.py)

후보는 지시문 문구(common, steps[].say)만 바꿀 수 있다 — 단계 id·순서·when·tools·always_tools·name이 바뀌면 거부한다
(절차·권한 변경은 지시문 층이 아니라 코드 층이다).
"""

from __future__ import annotations

import difflib
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.agent.protocol import PROTOCOL_DIR, load_protocol
from csagent.config import ROOT

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


class CandidateError(ValueError):
    pass


def _abs(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else ROOT / p


def new(case_dir: Path, issue: int, root: Path) -> Path:
    report = case_dir / "triage.md"
    if not report.exists():
        raise CandidateError(f"triage 보고서가 없다: {report}")
    from scripts.triage_save import parse
    head, _ = parse(report.read_text(encoding="utf-8"))
    if not 0 <= issue < len(head["issues"]):
        raise CandidateError(f"issue 번호 범위 밖: {issue} (0~{len(head['issues']) - 1})")
    item = head["issues"][issue]
    if item["layer"] != "prompt":
        raise CandidateError(f"지시문 층 문제가 아니다 (layer={item['layer']}) — APO 대상 아님")
    out = root / "runs" / "apo" / f"{datetime.now():%Y%m%d_%H%M%S}_{case_dir.name}_i{issue}"
    out.mkdir(parents=True)
    (out / "source.json").write_text(json.dumps({"case_dir": str(case_dir), "case": head["case"], "issue": issue,
                                                 **item}, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def check_structure(original: dict, candidate: dict) -> None:
    """문구 외의 것이 바뀌었나."""
    for key in ("name", "always_tools"):
        if original.get(key) != candidate.get(key):
            raise CandidateError(f"{key}가 바뀌었다 — 후보는 문구(common·say)만 바꾼다")
    ok, ck = original.get("steps") or [], candidate.get("steps") or []
    if [s["id"] for s in ok] != [s.get("id") for s in ck]:
        raise CandidateError("단계 id·순서가 바뀌었다 — 절차 변경은 코드 층")
    for a, b in zip(ok, ck):
        for key in ("when", "tools"):
            if (a.get(key) or None) != (b.get(key) or None):
                raise CandidateError(f"단계 {a['id']}의 {key}가 바뀌었다 — 절차·권한 변경은 코드 층")
    extra = set(candidate) - set(original)
    if extra:
        raise CandidateError(f"없던 키: {', '.join(sorted(extra))}")


def add(exp: Path, cand: str, protocol: str, text: str) -> Path:
    if not (exp / "source.json").exists():
        raise CandidateError(f"실험 폴더가 아니다 (source.json 없음): {exp}")
    if not _NAME.match(cand):
        raise CandidateError(f"후보 이름은 영문 소문자·숫자·_-: {cand}")
    orig_path = PROTOCOL_DIR / f"{protocol}.yaml"
    if not orig_path.exists():
        raise CandidateError(f"프로토콜 없음: {protocol}")
    original_text = orig_path.read_text(encoding="utf-8")
    try:
        candidate = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise CandidateError(f"YAML 오류: {e}") from None
    if not isinstance(candidate, dict):
        raise CandidateError("프로토콜 파일 전체(YAML 매핑)를 넣는다")
    check_structure(yaml.safe_load(original_text), candidate)
    if text.strip() == original_text.strip():
        raise CandidateError("원본과 같다")
    with tempfile.TemporaryDirectory() as tmp:   # 실제 로더로 읽혀야 한다 (when 어휘 검사 포함)
        (Path(tmp) / f"{protocol}.yaml").write_text(text, encoding="utf-8")
        try:
            load_protocol(protocol, Path(tmp))
        except Exception as e:   # noqa: BLE001 — 로더가 내는 모든 오류를 후보 거절 사유로
            raise CandidateError(f"로더가 읽지 못한다: {type(e).__name__}: {e}") from None
    d = exp / cand
    d.mkdir(exist_ok=True)
    (d / f"{protocol}.yaml").write_text(text, encoding="utf-8")
    diff = difflib.unified_diff(original_text.splitlines(keepends=True), text.splitlines(keepends=True),
                                f"a/{protocol}.yaml", f"b/{protocol}.yaml")
    (d / "diff.patch").write_text("".join(diff), encoding="utf-8")
    return d


def write_doc(exp: Path, rel: str, text: str) -> Path:
    if not (exp / "source.json").exists():
        raise CandidateError(f"실험 폴더가 아니다 (source.json 없음): {exp}")
    if not text.strip():
        raise CandidateError("내용이 비었다")
    path = exp / rel
    if not path.parent.exists():
        raise CandidateError(f"후보가 없다: {path.parent.name} (먼저 add)")
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = sys.argv[1:]
    try:
        if args[:1] == ["new"] and len(args) == 3:
            print(new(_abs(args[1]), int(args[2]), ROOT))
        elif args[:1] == ["add"] and len(args) == 4:
            d = add(_abs(args[1]), args[2], args[3], sys.stdin.read())
            print(f"후보 저장: {d}\n{(d / 'diff.patch').read_text(encoding='utf-8')}")
        elif args[:1] == ["note"] and len(args) == 3:
            print(f"저장: {write_doc(_abs(args[1]), f'{args[2]}/rationale.md', sys.stdin.read())}")
        elif args[:1] == ["proposal"] and len(args) == 2:
            print(f"저장: {write_doc(_abs(args[1]), 'proposal.md', sys.stdin.read())}")
        else:
            sys.exit(__doc__)
    except CandidateError as e:
        sys.exit(f"저장 안 함: {e}")


if __name__ == "__main__":
    main()
