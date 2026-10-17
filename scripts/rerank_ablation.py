"""리랭커 on/off 검색 비교 — 평가 로그의 실제 고객 발화로 에이전트와 같은 질의·필터를 재구성해 정답 문서(gold_docs) 적중을 본다.

  python scripts/rerank_ablation.py [--runs 'runs/2026101*'] [--k 4]

질의 = 직전 고객 발화 + 현재 고객 발화 (agent/loop.py `_retrieve`와 동일), 필터 = 시나리오 audience·companies.
1차 후보(hybrid RRF)는 한 번만 뽑아 on/off 양쪽에 같이 쓴다 → 차이는 리랭크 유무뿐.
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from csagent.evaluation.scenario import load_scenarios  # noqa: E402
from csagent.factory import build_retriever  # noqa: E402
from csagent.retrieval.filters import SearchFilter  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--runs", default=str(ROOT / "runs/2026101*"))
p.add_argument("--k", type=int, default=4)
args = p.parse_args()

scen = {s.id: s for d in ("scenarios", "scenarios_heldout") for s in load_scenarios(ROOT / d)}
heldout = {s.id for s in load_scenarios(ROOT / "scenarios_heldout")}

# (시나리오, 대화) → 질의 목록. 같은 대화가 여러 실행에 반복되면 하나로
convs = defaultdict(set)
for f in glob.glob(args.runs + "/results.jsonl"):
    for line in open(f):
        r = json.loads(line)
        sc = scen.get(r["scenario_id"])
        if sc is None or not sc.expect.gold_docs:
            continue
        cust = [t for role, t in r["turns"] if role == "customer"]
        queries = tuple(" ".join(cust[max(i - 1, 0):i + 1]) for i in range(len(cust)))
        convs[sc.id].add(queries)

ret = build_retriever(True)
cache = {}


def doc_ids(hits):
    return [h.chunk_id.split("#")[0] for h in hits]


def run(sc, q):
    key = (sc.id, q)
    if key not in cache:
        flt = SearchFilter(audience=sc.context.audience, companies=sc.context.companies)
        cands = ret._first_stage(q, flt.to_expr(), "hybrid")
        scores = ret.reranker.rerank(q, [h.display_text() for h in cands]) if cands else []
        rr = [h for h, _ in sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)]
        cache[key] = (doc_ids(cands), doc_ids(rr))
    return cache[key]


def rank(ids, gold):
    return next((i + 1 for i, d in enumerate(ids) if d in gold), None)


tot = defaultdict(lambda: defaultdict(int))
for sid, cs in sorted(convs.items()):
    sc = scen[sid]
    gold = set(sc.expect.gold_docs)
    for queries in cs:
        hit = {"off": False, "on": False}
        best = {"off": None, "on": None}
        first = {"off": None, "on": None}  # 정답이 처음 top-k에 든 고객 턴 번호
        for ti, q in enumerate(queries):
            off, on = run(sc, q)
            for name, ids in (("off", off), ("on", on)):
                r = rank(ids, gold)
                if r is not None and (best[name] is None or r < best[name]):
                    best[name] = r
                if r is not None and r <= args.k and first[name] is None:
                    first[name] = ti
                hit[name] |= r is not None and r <= args.k
        if first["off"] != first["on"]:
            tot["dev" if sid not in heldout else "heldout"]["first_diff"] += 1
            print(f"  첫 적중 턴 {sid}: off {first['off']} / on {first['on']}")
        grp = "heldout" if sid in heldout else "dev"
        t = tot[grp]
        t["convs"] += 1
        t["hit_off"] += hit["off"]
        t["hit_on"] += hit["on"]
        t["only_on"] += hit["on"] and not hit["off"]
        t["only_off"] += hit["off"] and not hit["on"]
        if hit["on"] != hit["off"]:
            print(f"  차이 {sid}: off 최고순위 {best['off']} / on {best['on']}  정답 {sorted(gold)}")
    print(f"{sid}: 대화 {len(cs)}")

for grp, t in tot.items():
    n = t["convs"]
    print(f"[{grp}] 대화 {n}건 — 정답 top{args.k} 적중(대화 중 한 번이라도): "
          f"리랭크 끔 {t['hit_off']}/{n} · 켬 {t['hit_on']}/{n}  (켬에서만 {t['only_on']}, 끔에서만 {t['only_off']})")
print("고유 질의", len(cache))

# 질의 단위: 1차 후보 안에 정답이 있는 질의만 (없으면 리랭크로도 못 살림)
q_tot = defaultdict(int)
for (sid, q), (off, on) in cache.items():
    gold = set(scen[sid].expect.gold_docs)
    if rank(off, gold) is None:
        q_tot["no_cand"] += 1
        continue
    ro, rn = rank(off, gold), rank(on, gold)
    q_tot["n"] += 1
    q_tot["off"] += ro <= args.k
    q_tot["on"] += rn <= args.k
    q_tot["on1"] += rn == 1
    q_tot["off1"] += ro == 1
print(f"[질의 단위] 1차 후보에 정답 있는 질의 {q_tot['n']}건 (없음 {q_tot['no_cand']}) — "
      f"top{args.k} 적중 끔 {q_tot['off']} · 켬 {q_tot['on']} / 1위 끔 {q_tot['off1']} · 켬 {q_tot['on1']}")
