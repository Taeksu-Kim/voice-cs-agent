"""근거 판정 v4 — 2단계 (docs/07 8절): ① 발화에서 사실 주장만 뽑고 ② 주장마다 근거와 대조한다.

v3(한 번에 판정)의 문제: 「무엇이 사실 주장인가」를 고를 때부터 업무 지시문을 보고 있어 지시 준수가 판정에 섞였다
(지시문만 바꿔도 질문 「상담원 연결해 드릴까요?」가 근거 없음이 됨). 주장 하나만 내고 근거 인용이 없어 판정을 검증할 수 없었다.

- 1단계(주장 뽑기)는 상담원 발화와 직전 고객 말만 본다 — 근거·지시문을 안 주므로 지시문을 바꿔도 주장 목록이 같다
- 2단계(대조)는 주장마다 supported / unsupported / contradicted + 출처 + 근거 원문 인용
- 코드 대조 (LLM 말을 그대로 믿지 않는다):
  · 주장의 span이 실제 발화에 없으면 버린다
  · supported인데 근거 인용이 실제 문서·지시문에 없거나, 출처 도구가 성공하지 않았고 인용도 그 결과에 없으면 unverified (실패로 센다)
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any, Optional

from csagent.llm import ChatLLM, parse_json

_CLAIMS = """\
너는 고객센터 상담원 발화에서 확인할 수 있는 사실 주장을 뽑는다. 맞는지 틀린지는 판정하지 않는다.
사실 주장 = 정책·수치·날짜·기간·절차·연락처·고객이 할 수 있는 일이나 해야 할 일에 대한 안내,
조회·처리 결과 보고(「배송중이에요」「보내 드렸어요」「정지했어요」「인증 완료됐어요」).
뽑지 않는 것: 인사·공감·사과, 질문과 제안(「~해 드릴까요?」「~하시겠어요?」), 앞으로 하겠다는 진행 안내
(「확인해 볼게요」「보내 드릴게요」), 고객이 한 말을 되풀이한 것,
상담원 자신이 이 통화에서 할 수 있다·없다는 말(「여기서는 안내해 드릴 수 없어요」「제가 확인하기 어려워요」).
단 회사·서비스 차원의 정책은 뽑는다 (「토스에서는 처리할 수 없어요」「앱에서만 신청돼요」).
질문·제안 안에 사실이 들어 있으면 그 사실만 뽑는다 (「CVC가 필요한데 진행할까요?」 → 「카드 등록에 CVC가 필요하다」).
한 문장에 사실이 여럿이면 나눈다.

[직전 고객 말]
{customer}

[상담원 발화]
{say}

[출력] JSON 하나: {{"claims": [{{"span": "발화에서 글자 그대로 옮긴 부분", "claim": "주장을 한 문장으로",
"type": "결과" 또는 "안내"}}]}} — 사실 주장이 없으면 빈 목록"""

_VERIFY = """\
너는 고객센터 상담의 사실 검증자다. 아래 [주장]마다 [근거]로 뒷받침되는지 판정한다.
상담원이 업무 지시대로 행동했는지는 보지 않는다 — 주장 내용이 근거에 있는지만 본다.
- supported: 근거에 같은 내용이 있다 (표현이 달라도 된다). evidence에 근거 원문을 글자 그대로 옮긴다
- contradicted: 근거와 어긋난다. evidence에 어긋나는 근거 원문
- unsupported: 근거 어디에도 없다. evidence는 빈 문자열
source: "doc"(검색 문서) | "instruction"(업무 지시) | "tool:<도구 이름>"(도구 결과) | ""(없음)
조회·처리 결과 보고는 [근거: 도구 결과]에 그 도구의 성공 결과(ok: true)가 있고 내용이 맞으면 supported, source는 "tool:<도구 이름>".
업무 지시에 적힌 절차·조건·안내는 회사가 정한 사실로 인정한다.

[근거: 검색 문서]
{docs}

[근거: 도구 결과]
{tools}

[근거: 업무 지시]
{instructions}

[상담원 발화 전체 — 맥락]
{say}

[주장]
{claims}

[출력] JSON 하나: {{"verdicts": [{{"i": 주장 번호, "verdict": "supported" | "unsupported" | "contradicted",
"source": "...", "evidence": "..."}}]}}"""

CLAIMS_SCHEMA = {"type": "object", "properties": {"claims": {"type": "array", "items": {
    "type": "object", "properties": {"span": {"type": "string"}, "claim": {"type": "string"},
                                     "type": {"enum": ["결과", "안내"]}},
    "required": ["span", "claim", "type"]}}}, "required": ["claims"]}
VERIFY_SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {
    "type": "object", "properties": {"i": {"type": "integer"},
                                     "verdict": {"enum": ["supported", "unsupported", "contradicted"]},
                                     "source": {"type": "string"}, "evidence": {"type": "string"}},
    "required": ["i", "verdict", "source", "evidence"]}}}, "required": ["verdicts"]}

FAILING = ("unsupported", "contradicted", "unverified")
_STRIP = re.compile(r"[\s\W_]+")


def _norm(text: str) -> str:
    return _STRIP.sub("", text)


def quoted_in(quote: str, source: str) -> bool:
    """인용이 원문에 있나 — 공백·문장부호 차이는 무시하고, 긴 인용은 70%가 이어서 맞으면 인정 (LLM이 조금 줄여 옮김)."""
    q, s = _norm(quote), _norm(source)
    if len(q) < 4:
        return False
    if q in s:
        return True
    m = difflib.SequenceMatcher(None, q, s, autojunk=False).find_longest_match(0, len(q), 0, len(s))
    return len(q) >= 15 and m.size >= 0.7 * len(q)


def _quoted(evidence: str, source: str) -> bool:
    """LLM이 중간을 「...」로 줄여 옮기면 조각마다 확인한다."""
    parts = [p for p in re.split(r"\.{3}|…", evidence) if len(_norm(p)) >= 4]
    return bool(parts) and all(quoted_in(p, source) for p in parts)


def _tool_backs(name: str, evidence: str, tools: list[dict[str, Any]]) -> bool:
    """출처 도구가 성공했거나, 실패했어도 근거 인용이 그 결과에 있다 (「처리 중 오류가 났어요」는 실패 결과가 근거)."""
    mine = [t for t in tools if t.get("name") == name]
    return any(t.get("ok") for t in mine) or bool(mine) and _quoted(evidence, json.dumps(mine, ensure_ascii=False))


def extract_claims(llm: ChatLLM, say: str, customer: str = "") -> Optional[list[dict[str, Any]]]:
    """1단계. 파싱 실패면 None. 발화에 없는 span은 버린다."""
    try:
        res = parse_json(llm.chat([{"role": "user", "content": _CLAIMS.format(customer=customer or "(없음)", say=say)}],
                                  max_tokens=500, json_mode=True, schema=CLAIMS_SCHEMA).text)
        claims = res["claims"]
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return [c for c in claims if isinstance(c, dict) and c.get("span") and _norm(c["span"]) in _norm(say)]


def verify_claims(llm: ChatLLM, claims: list[dict[str, Any]], *, say: str, docs: list[str],
                  tools: list[dict[str, Any]], instructions: str) -> Optional[list[dict[str, Any]]]:
    """2단계. 주장마다 {span, claim, type, verdict, source, evidence, note}. 파싱 실패면 None."""
    prompt = _VERIFY.format(
        docs="\n\n".join(f"[문서{i + 1}] {d}" for i, d in enumerate(docs)) or "(없음)",
        tools=json.dumps(tools, ensure_ascii=False) if tools else "(없음)",
        instructions=instructions or "(없음)", say=say,
        claims="\n".join(f"{i}. {c['claim']} (발화: 「{c['span']}」)" for i, c in enumerate(claims)))
    try:
        verdicts = {v["i"]: v for v in parse_json(llm.chat([{"role": "user", "content": prompt}], max_tokens=800,
                                                           json_mode=True, schema=VERIFY_SCHEMA).text)["verdicts"]}
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    corpus = "\n".join([*docs, instructions, json.dumps(tools, ensure_ascii=False)])   # 출처 표기가 틀려도 인용만 맞으면 인정
    out = []
    for i, c in enumerate(claims):
        v = verdicts.get(i)
        if v is None:
            out.append({**c, "verdict": "unverified", "source": "", "evidence": "", "note": "판정 누락"})
            continue
        verdict, source, evidence, note = v["verdict"], v.get("source", ""), v.get("evidence", ""), ""
        if verdict == "supported":
            if source.startswith("tool:"):
                if not _tool_backs(source[5:], evidence, tools):
                    verdict, note = "unverified", f"{source[5:]} 결과가 뒷받침 안 함"
            elif not _quoted(evidence, corpus):
                verdict, note = "unverified", "근거 인용이 원문에 없음"
        out.append({**c, "verdict": verdict, "source": source, "evidence": evidence, "note": note})
    return out


def judge_turn(llm: ChatLLM, *, say: str, customer: str, docs: list[str], tools: list[dict[str, Any]],
               instructions: str) -> dict[str, Any]:
    """턴 하나: {unsupported (파싱 실패면 None), claim (실패 주장 요약), claims (주장별 기록)}."""
    claims = extract_claims(llm, say, customer)
    if claims is None:
        return {"unsupported": None, "claim": "", "claims": [], "stage": "extract"}
    if not claims:
        return {"unsupported": False, "claim": "", "claims": []}
    judged = verify_claims(llm, claims, say=say, docs=docs, tools=tools, instructions=instructions)
    if judged is None:
        return {"unsupported": None, "claim": "", "claims": claims, "stage": "verify"}
    bad = [c for c in judged if c["verdict"] in FAILING]
    return {"unsupported": bool(bad), "claim": "; ".join(f"「{c['span']}」 {c['verdict']}" for c in bad),
            "claims": judged}
