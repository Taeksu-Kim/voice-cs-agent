"""검색 점검용 CLI.

    python scripts/search.py "돈을 잘못 보냈어요" [--mode hybrid|dense|sparse] [--k 5] [--rerank]
        [--audience consumer|merchant|seller] [--company 토스뱅크 ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.factory import build_retriever
from csagent.retrieval.filters import SearchFilter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="+")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "sparse"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--audience", choices=["consumer", "merchant", "seller"])
    ap.add_argument("--company", nargs="*", default=[])
    args = ap.parse_args()

    retriever = build_retriever(args.rerank)
    flt = SearchFilter(audience=args.audience, companies=tuple(args.company))
    for q in args.queries:
        print(f"\n### {q}  [{args.mode}{' +rerank' if args.rerank else ''}]")
        hits = retriever.search(q, args.k, flt=flt, mode=args.mode)
        for i, h in enumerate(hits, 1):
            sec = f" > {h.section}" if h.section else ""
            tags = f"{h.meta.get('audience')}/{','.join(h.meta.get('domains', []))}"  # 도메인은 표시만
            print(f"{i}. {h.score:.4f} [{tags}] {h.chunk_id}  {h.title}{sec}")


if __name__ == "__main__":
    main()
