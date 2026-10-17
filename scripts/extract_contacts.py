"""청크 본문의 연락처를 LLM으로 뽑는다: 어느 기관 번호인지, 어떤 경우에 거기로 안내하는지.

정규식으로는 「누구 번호인지·언제 쓰는 번호인지」를 못 가린다 (faq:6705 「토스뱅크 통장이 지급정지 되었다면 1661-7654」,
faq:236 카드사 번호 목록). 정규식은 LLM에 보낼 후보 청크를 고르는 데만 쓴다.
LLM이 낸 번호는 본문에 숫자 그대로 있어야 남긴다 (지어낸 번호 차단).
토스 계열 번호의 주인은 원문으로 확인한 표(taxonomy.TOSS_CONTACTS)가 정한다 — 문서가 「고객센터(1661-7654)」처럼
주인을 안 밝히면 LLM이 추측한다 (실측: faq:4283을 「토스 고객센터」로). LLM은 조건(when)·종류만 맡는다.

    python scripts/extract_contacts.py            # data/processed/contacts.jsonl
    python scripts/extract_contacts.py --limit 5  # 몇 건만 시험

필요: LLM 서버(LLM_URL). 결과는 chunk_id + 본문 해시로 남겨, 청크가 바뀌면 다시 뽑아야 함을 알 수 있게 한다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csagent.config import load_settings
from csagent.agent.guardrails.contacts import text_hash
from csagent.knowledge.taxonomy import TOSS_CONTACTS
from csagent.llm import OpenAICompatLLM, parse_json

CANDIDATE = re.compile(r"\d{2,4}[-‐ .)]\s?\d{3,4}[- .]\d{4}|\d{4}-\d{4}|전화|고객센터|콜센터|연락처")

PROMPT = """\
아래 고객센터 문서에서 **전화번호**를 모두 찾아 JSON으로 정리하라. 문서에 없는 번호·기관을 만들지 않는다.

각 번호마다:
- phone: 문서에 적힌 그대로
- kind: 전화 / 팩스 / 기타 (문서가 팩스라고 하면 팩스)
- org: 그 번호의 주인 기관 이름 (예: 토스 고객센터, 토스뱅크 고객센터, 신한카드, KCB 올크레딧). 문서가 밝힌 이름 그대로.
  「고객센터」처럼 어느 기관인지 문서에 안 나오면 짐작하지 말고 ""
- when: 문서가 이 번호로 연락하라고 한 경우·용도 (예: 「토스뱅크 통장이 지급정지된 경우」, 「카드 분실 신고」). 조건이 없으면 ""
- hours: 운영 시간이 적혀 있으면 그대로, 없으면 ""
번호가 없으면 contacts를 빈 목록으로.

[문서]
{doc}"""

SCHEMA = {
    "type": "object",
    "properties": {"contacts": {"type": "array", "items": {
        "type": "object",
        "properties": {"phone": {"type": "string"}, "kind": {"type": "string", "enum": ["전화", "팩스", "기타"]},
                       "org": {"type": "string"}, "when": {"type": "string"}, "hours": {"type": "string"}},
        "required": ["phone", "kind", "org", "when", "hours"]}}},
    "required": ["contacts"],
}


def digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def grounded(contact: dict, text: str) -> bool:
    """번호의 숫자 묶음이 본문에 그 순서로 있는가 — 묶음 사이 구분자(-·공백·원문 오타 →)는 허용, 앞뒤에 숫자가 붙으면 안 됨.
    짧은 대표번호(1332 금감원, 126 국세청)도 받는다."""
    groups = re.findall(r"\d+", contact["phone"])
    if not groups or sum(map(len, groups)) < 3:
        return False
    pat = r"(?<!\d)" + r"\s*[-‐.→)]?\s*".join(groups) + r"(?!\d)"
    return re.search(pat, text) is not None


def canonical(contact: dict) -> dict:
    """토스 계열 번호면 주인을 표에서 정한다. 번호는 숫자 묶음을 -로 이어 표준화 (원문 표기는 phone_raw)."""
    phone = "-".join(re.findall(r"\d+", contact["phone"]))
    out = {**contact, "phone": phone, "phone_raw": contact["phone"]}
    if phone in TOSS_CONTACTS:
        out.update(service=TOSS_CONTACTS[phone], verified=True)
    else:
        named = [s for s in sorted(set(TOSS_CONTACTS.values()), key=len, reverse=True)
                 if s in contact["org"].replace(" ", "")]
        # 토스 계열 이름이 붙었는데 표에 없는 번호 = 원문 오타이거나 확인 안 된 번호 (faq:6553 「토스뱅크 1661-7633」)
        out.update(service=named[0] if named else "", verified=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    s = load_settings()
    llm = OpenAICompatLLM(s.llm_url, s.llm_model, timeout=300)
    chunks = [json.loads(l) for l in (s.processed_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    cands = [c for c in chunks if CANDIDATE.search(c["title"] + "\n" + c["text"])]
    if args.limit:
        cands = cands[:args.limit]
    print(f"청크 {len(chunks)} 중 후보 {len(cands)}")

    out = s.processed_dir / ("contacts.sample.jsonl" if args.limit else "contacts.jsonl")
    t0, dropped, found = time.time(), [], 0
    with out.open("w", encoding="utf-8") as f:
        for i, c in enumerate(cands, 1):
            doc = f"{c['title']}\n{c['text']}"
            r = llm.chat([{"role": "user", "content": PROMPT.format(doc=doc)}],
                         max_tokens=1500, schema=SCHEMA)
            items = parse_json(r.text).get("contacts", [])
            keep = [canonical(x) for x in items if grounded(x, doc)]
            dropped += [(c["chunk_id"], x["phone"]) for x in items if not grounded(x, doc)]
            found += len(keep)
            if keep:
                f.write(json.dumps({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "hash": text_hash(c["title"], c["text"]),
                                    "contacts": keep}, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(f"  {i}/{len(cands)} 번호 {found} — {time.time() - t0:.0f}s", flush=True)
    print(f"완료: 번호 {found}, 본문에 없어 버림 {len(dropped)} {dropped[:10]} — {time.time() - t0:.0f}s → {out}")


if __name__ == "__main__":
    main()
