"""근거 판정 프롬프트를 바꿨을 때 참조 라벨과의 일치가 유지되는지 — 저장된 판정 입력을 지금 프롬프트로 다시 판정한다.

    python scripts/grounded_agreement.py [--items labels/rejudge_20261017_192929/items.jsonl] [--labels labels/judge_v1]

입력은 rejudge가 남긴 items.jsonl (발화·검색 문서·도구 결과·지시문 그대로), 라벨은 judge_v1의 labels.jsonl·sample_meta.jsonl.
무작위 세트(일치율 계산용)와 표적 세트(알려진 오판 후보)를 따로 센다. 라벨은 Opus 참조 라벨이다 — 사람 라벨이 아니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.config import ROOT
from csagent.evaluation.scoring import _GROUNDED
from csagent.factory import build_llm
from csagent.llm import parse_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(ROOT / "labels" / "rejudge_20261017_192929" / "items.jsonl"))
    ap.add_argument("--labels", default=str(ROOT / "labels" / "judge_v1"))
    args = ap.parse_args()

    read = lambda p: [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]
    labels = {x["key"]: x for x in read(Path(args.labels) / "labels.jsonl")}
    meta = {x["key"]: x for x in read(Path(args.labels) / "sample_meta.jsonl")}
    items = {x["key"]: x for x in read(args.items) if x["kind"] == "grounded"}
    keys = [k for k in labels if meta.get(k, {}).get("kind") == "grounded" and k in items]
    llm = build_llm()
    rows = []
    for k in keys:
        it = items[k]
        res = parse_json(llm.chat([{"role": "user", "content": _GROUNDED.format(
            docs=it["docs"], tools=it["tools"], instructions=it["instructions"], say=it["say"])}],
            max_tokens=200, json_mode=True).text)
        now = not str(res.get("unsupported", "")).upper().startswith("Y")
        rows.append({"key": k, "set": meta[k]["set"], "label": labels[k]["label"], "conf": labels[k]["confidence"],
                     "before": meta[k]["judge_pass"], "now": now, "claim": res.get("claim", ""), "say": it["say"]})
    for s in ("random", "targeted"):
        rs = [r for r in rows if r["set"] == s]
        if rs:
            print(f"[{s}] {len(rs)}건 — 라벨 일치: 이전 프롬프트 {sum(r['before'] == r['label'] for r in rs)} · "
                  f"지금 프롬프트 {sum(r['now'] == r['label'] for r in rs)}")
    for r in rows:
        if r["before"] != r["now"] or r["now"] != r["label"]:
            mark = "✓" if r["now"] == r["label"] else "✗"
            print(f"  {mark} {r['key']} [{r['set']}] 라벨={'통과' if r['label'] else '실패'}({r['conf']}) "
                  f"이전={'통과' if r['before'] else '실패'} 지금={'통과' if r['now'] else '실패'} 「{r['claim'] or r['say'][:60]}」")


if __name__ == "__main__":
    main()
