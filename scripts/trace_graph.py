"""통화 하나를 mermaid 흐름도로 — 노드 id가 좌표(t2_s1_gate = trace at "t2.s1.gate")다 (csagent/tracegraph.py).

    python scripts/trace_graph.py runs/20261020_071342_C s3_b0_defect:1     # 평가 기록: 시나리오[:반복]
    python scripts/trace_graph.py logs/20261019 3f2a9c01be4d                # 서버 통화 기록: call_id
    python scripts/trace_graph.py runs/20261020_071342_C s3_b0_defect:1 --out graph.md

--out을 주면 ```mermaid 블록이 든 md로 (웹 md 뷰어·GitHub에서 그림으로 보인다), 없으면 화면에.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.config import ROOT
from csagent.tracegraph import to_mermaid


def find(source: Path, key: str) -> tuple[dict, str, list[str]]:
    """(기록 한 줄, 제목, 실패 항목)."""
    if (source / "results.jsonl").exists():
        sid, _, rep = key.partition(":")
        for line in (source / "results.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            if row["scenario_id"] == sid and (not rep or row["repeat"] == int(rep)):
                failed = [f"{c['name']}: {c.get('detail') or ''}" for c in row["checks"] if c["passed"] is False]
                return row, f"{source.name} {sid} r{row['repeat']} · {row['disposition']}", failed
    for line in (source / "calls.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        if row["call_id"] == key:
            return row, f"{source.name} {key} · {row.get('disposition')}", []
    raise SystemExit(f"없음: {source} {key}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="runs/<실행> 또는 logs/<날짜>")
    ap.add_argument("key", help="평가: 시나리오[:반복] / 서버: call_id")
    ap.add_argument("--out")
    args = ap.parse_args()
    source = Path(args.source) if Path(args.source).is_absolute() else ROOT / args.source
    row, title, failed = find(source, args.key)
    graph = to_mermaid(row["trace"], title, failed)
    if args.out:
        Path(args.out).write_text(f"# {title}\n\n```mermaid\n{graph}```\n", encoding="utf-8")
        print(f"→ {args.out}")
    else:
        print(graph)


if __name__ == "__main__":
    main()
