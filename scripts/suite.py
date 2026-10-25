"""케이스 모음 — 업무별로 재사용하는 턴 단위 회귀 세트 (docs/10 6절). 실험할 때마다 쌓인다.

    python scripts/suite.py list [<업무>]
    python scripts/suite.py add-triage <triage 케이스 폴더> <issue 번호> <업무> [--id <이름>]   # 재현 기준 → defect
    python scripts/suite.py add-variant <apo 실험 폴더> <테스트 이름> <업무> --kind guard|defect [--id <이름>] [--note …]
    python scripts/suite.py promote <업무> <id>                                                # defect → guard (반영·verify 뒤)

항목 = 재현 명세 하나 (기록된 통화의 한 지점 + 술어). 종류:
- defect: 알려진 결함 — 지금 지시문에서 실패한다. 고치면 통과해야 한다
- guard: 지금 잘하는 행동 — 계속 통과해야 한다 (고친 defect는 verify 뒤 guard로)
apo-test 계획(tests.yaml)에 `suite: <업무>`를 쓰면 그 업무의 항목이 모두 guard 테스트로 들어간다 (scripts/apo_test.py).

저장: runs/suite/<업무>.yaml (통화 기록을 참조하므로 runs/ 아래 — 저장소에 올리지 않는다)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.config import ROOT

_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,60}$")
KINDS = ("defect", "guard")


class SuiteError(ValueError):
    pass


def _path(root: Path, task: str) -> Path:
    if not _ID.match(task):
        raise SuiteError(f"업무 이름: {task}")
    return root / "runs" / "suite" / f"{task}.yaml"


def load(root: Path, task: str) -> list[dict]:
    p = _path(root, task)
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or []) if p.exists() else []


def _save(root: Path, task: str, items: list[dict]) -> None:
    p = _path(root, task)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(items, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _spec(path: Path) -> dict:
    if not path.exists():
        raise SuiteError(f"명세 없음: {path}")
    d = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else yaml.safe_load(path.read_text(encoding="utf-8"))
    if not d.get("fail_if"):
        raise SuiteError(f"fail_if 없는 명세: {path}")
    return {k: v for k, v in d.items() if v is not None and k != "repeat"}


def add(root: Path, task: str, item: dict) -> dict:
    items = load(root, task)
    if not _ID.match(item["id"]):
        raise SuiteError(f"id: {item['id']}")
    if item["kind"] not in KINDS:
        raise SuiteError(f"kind는 {' | '.join(KINDS)}")
    if any(x["id"] == item["id"] for x in items):
        raise SuiteError(f"이미 있는 id: {item['id']}")
    same = [x["id"] for x in items if x["spec"] == item["spec"]]
    if same:
        raise SuiteError(f"같은 명세가 이미 있다: {same[0]}")
    items.append(item)
    _save(root, task, items)
    return item


def add_triage(root: Path, case_dir: Path, issue: int, task: str, item_id: str | None = None) -> dict:
    from scripts.triage_save import parse
    head, _ = parse((case_dir / "triage.md").read_text(encoding="utf-8"))
    it = head["issues"][issue]
    if it["confidence"] != "reproduced" or not it.get("spec"):
        raise SuiteError("재현된 문제만 넣는다 (가설은 재현 명세가 없다)")
    spec = _spec(root / it["spec"] if not Path(it["spec"]).is_absolute() else Path(it["spec"]))
    return add(root, task, {"id": item_id or f"{case_dir.name}_i{issue}".lower(), "kind": "defect", "spec": spec,
                            "origin": f"triage {case_dir.name} issue {issue}: {it['summary']}",
                            "baseline": it.get("baseline", ""), "added": date.today().isoformat()})


def add_variant(root: Path, exp: Path, test: str, task: str, kind: str, item_id: str | None = None,
                note: str = "") -> dict:
    spec = _spec(exp / "tests" / f"{test}.yaml")
    return add(root, task, {"id": item_id or f"{exp.name[:15]}_{test}".lower(), "kind": kind, "spec": spec,
                            "origin": f"apo-test {exp.name} {test}" + (f": {note}" if note else ""),
                            "added": date.today().isoformat()})


def promote(root: Path, task: str, item_id: str) -> dict:
    items = load(root, task)
    hit = next((x for x in items if x["id"] == item_id), None)
    if hit is None:
        raise SuiteError(f"없는 id: {item_id}")
    if hit["kind"] != "defect":
        raise SuiteError("defect만 guard로 바꾼다")
    hit["kind"], hit["promoted"] = "guard", date.today().isoformat()
    _save(root, task, items)
    return hit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list"); s.add_argument("task", nargs="?")
    s = sub.add_parser("add-triage"); s.add_argument("case_dir"); s.add_argument("issue", type=int); s.add_argument("task")
    s.add_argument("--id")
    s = sub.add_parser("add-variant"); s.add_argument("exp"); s.add_argument("test"); s.add_argument("task")
    s.add_argument("--kind", required=True, choices=KINDS); s.add_argument("--id"); s.add_argument("--note", default="")
    s = sub.add_parser("promote"); s.add_argument("task"); s.add_argument("id")
    a = ap.parse_args()
    ab = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
    try:
        if a.cmd == "list":
            tasks = [a.task] if a.task else sorted(p.stem for p in (ROOT / "runs" / "suite").glob("*.yaml"))
            for t in tasks:
                print(f"[{t}]")
                for x in load(ROOT, t):
                    print(f"  {x['kind']:6s} {x['id']:40s} {x['spec']['at']:6s} {x['spec'].get('mode', 'turn'):8s} {x['origin'][:70]}")
        elif a.cmd == "add-triage":
            print("추가:", add_triage(ROOT, ab(a.case_dir), a.issue, a.task, a.id)["id"])
        elif a.cmd == "add-variant":
            print("추가:", add_variant(ROOT, ab(a.exp), a.test, a.task, a.kind, a.id, a.note)["id"])
        else:
            print("guard로:", promote(ROOT, a.task, a.id)["id"])
    except SuiteError as e:
        sys.exit(f"중단: {e}")


if __name__ == "__main__":
    main()
