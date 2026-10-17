"""판정기 vs 참조 라벨 일치도 (1-7 Judge 신뢰도). 기준은 labels/RUBRIC.md.

    python scripts/judge_agreement.py sample labels/rejudge_<시각> [--out labels/judge_v1]
    python scripts/judge_agreement.py score  labels/rejudge_<시각> [--out labels/judge_v1]

sample — 라벨 붙일 항목을 고른다.
  무작위 세트: 종류별 할당량을 판정기 다수결(통과/실패) 층으로 나눠 뽑는다 (실패가 드물어 그냥 뽑으면 실패 탐지를 못 잰다).
    층마다 표본 가중치(모집단 수 / 표본 수)를 남겨, 전체 일치율은 모집단 비율로 되돌려 계산한다.
  표적 세트: 재판정에서 뒤집힌 항목 + --include로 지정한 항목 (알려진 오판 후보). 오판 유형 분석용 — 일치율에 넣지 않는다.
  sample.jsonl(라벨러용 — 판정 결과·층 정보 없음)과 sample_meta.jsonl(채점용)을 따로 쓴다.
score — labels.jsonl과 판정기 다수결을 비교해 종류별 일치율, 가중 일치율, Cohen's κ, 실패 탐지 정밀도·재현율을 낸다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

QUOTA = {"mention": 10, "must_not": 6, "goal_resolved": 8, "false_completion": 4, "grounded": 12}   # 합 40
TARGETED_MAX = 10


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def majority(v: dict, k: int) -> bool | None:
    vs = [x for x in v["verdicts"][:k] if x is not None]
    return None if not vs else sum(vs) * 2 > len(vs)


def flipped(v: dict) -> bool:
    return len({x for x in v["verdicts"] if x is not None}) > 1


def sample(rejudge: Path, out: Path, k: int, include: list[str], seed: int) -> None:
    verdicts = {v["key"]: v for v in read_jsonl(rejudge / "verdicts.jsonl")}
    items = {i["key"]: i for i in read_jsonl(rejudge / "items.jsonl")}
    rng = random.Random(seed)
    strata = defaultdict(list)
    for key, v in verdicts.items():
        m = majority(v, k)
        if m is not None:
            strata[(v["kind"], m)].append(key)

    meta, quota = {}, dict(QUOTA)
    leftover = quota.get("false_completion", 0) - min(quota.get("false_completion", 0),
                                                      sum(len(strata[("false_completion", m)]) for m in (True, False)))
    quota["grounded"] += leftover   # false_completion 항목이 모자라면 grounded로 채운다
    quota["false_completion"] -= leftover
    for kind, n in quota.items():
        fails, passes = sorted(strata[(kind, False)]), sorted(strata[(kind, True)])
        n_fail = min(len(fails), n // 2)
        picks = [(s, rng.sample(pool, m)) for s, pool, m in
                 ((False, fails, n_fail), (True, passes, min(len(passes), n - n_fail)))]
        for judge_pass, keys in picks:
            pop = len(strata[(kind, judge_pass)])
            for key in keys:
                meta[key] = {"key": key, "set": "random", "kind": kind, "judge_pass": judge_pass,
                             "weight": pop / len(keys)}

    targeted = [key for key, v in sorted(verdicts.items()) if flipped(v) and key not in meta]
    targeted += [key for key in sorted(verdicts) if any(p in key for p in include) and key not in meta]
    for key in list(dict.fromkeys(targeted))[:TARGETED_MAX]:
        meta[key] = {"key": key, "set": "targeted", "kind": verdicts[key]["kind"],
                     "judge_pass": majority(verdicts[key], k), "weight": 0}

    out.mkdir(parents=True, exist_ok=True)
    keys = list(meta)
    rng.shuffle(keys)   # 라벨러가 층·세트 순서를 짐작하지 못하게
    write_jsonl(out / "sample.jsonl", [items[k_] for k_ in keys])
    write_jsonl(out / "sample_meta.jsonl", [meta[k_] for k_ in keys])
    n_rand = sum(m["set"] == "random" for m in meta.values())
    print(f"무작위 {n_rand} + 표적 {len(meta) - n_rand} = {len(meta)}건 → {out}")


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def score(rejudge: Path, out: Path, k: int, labels_file: str) -> None:
    verdicts = {v["key"]: v for v in read_jsonl(rejudge / "verdicts.jsonl")}
    meta = {m["key"]: m for m in read_jsonl(out / "sample_meta.jsonl")}
    labels = {l["key"]: l for l in read_jsonl(out / labels_file)}
    missing = [k_ for k_ in meta if k_ not in labels]
    if missing:
        print(f"라벨 없는 항목 {len(missing)}건 — 있는 것만 계산", file=sys.stderr)

    rows = [(meta[k_], majority(verdicts[k_], k), labels[k_]) for k_ in meta if k_ in labels]
    rand = [(m, j, l) for m, j, l in rows if m["set"] == "random"]

    print(f"{'종류':<18}{'표본':>5}{'일치':>6}{'일치율':>8}")
    by_kind = defaultdict(list)
    for m, j, l in rand:
        by_kind[m["kind"]].append(j == l["label"])
    for kind, xs in sorted(by_kind.items()):
        print(f"{kind:<18}{len(xs):>5}{sum(xs):>6}{sum(xs) / len(xs):>8.0%}")

    w_all = sum(m["weight"] for m, _, _ in rand)
    w_agree = sum(m["weight"] for m, j, l in rand if j == l["label"])
    print(f"\n무작위 {len(rand)}건: 표본 일치율 {sum(j == l['label'] for _, j, l in rand) / len(rand):.0%}"
          f" · 모집단 가중 일치율 {w_agree / w_all:.0%}")
    kp = kappa([(j, l["label"]) for _, j, l in rand])
    print(f"Cohen's κ (표본, 가중 없음): {kp:.2f}" if kp is not None else "κ 계산 불가")
    # 실패 탐지 — 양성 = 실패 (판정기가 실패라고 한 것 중 참조도 실패인 비율 / 참조 실패 중 판정기가 잡은 비율)
    tp = sum(m["weight"] for m, j, l in rand if not j and not l["label"])
    fp = sum(m["weight"] for m, j, l in rand if not j and l["label"])
    fn = sum(m["weight"] for m, j, l in rand if j and not l["label"])
    print(f"실패 탐지 (가중): 정밀도 {tp / (tp + fp):.0%} · 재현율 {tp / (tp + fn):.0%}" if tp + fp and tp + fn
          else "실패 탐지: 표본 부족")
    low = [l for _, _, l in rows if l.get("confidence") == "low"]
    print(f"참조 라벨 확신 낮음 {len(low)}건")

    print("\n불일치 (무작위·표적 모두)")
    for m, j, l in rows:
        if j != l["label"]:
            print(f"- [{m['set']}] {m['key']}  판정기 {'통과' if j else '실패'} / 참조 {'통과' if l['label'] else '실패'}"
                  f" ({l.get('confidence')}) — {l.get('reason', '')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "score"])
    ap.add_argument("rejudge", type=Path, help="scripts/rejudge.py 결과 폴더")
    ap.add_argument("--out", type=Path, default=Path("labels/judge_v1"))
    ap.add_argument("--k", type=int, default=3, help="다수결에 쓸 원래 순서 재판정 수 (rejudge --k와 같게)")
    ap.add_argument("--include", nargs="*", default=[], help="표적 세트에 넣을 항목 key 부분 문자열")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", default="labels.jsonl", help="score: --out 안의 라벨 파일 (판정 질문이 바뀌어 다시 붙인 판본 등)")
    args = ap.parse_args()
    if args.cmd == "sample":
        sample(args.rejudge, args.out, args.k, args.include, args.seed)
    else:
        score(args.rejudge, args.out, args.k, args.labels)


if __name__ == "__main__":
    main()
