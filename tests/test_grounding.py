"""근거 판정 v4 — 2단계 (csagent/evaluation/grounding.py)."""
import json

from csagent.evaluation.grounding import extract_claims, judge_turn, quoted_in
from csagent.llm import ScriptedLLM

SAY = "판매자가 평균 2영업일 안에 답해 줘요. 사진 첨부해서 접수해 드릴까요?"
DOCS = ["교환·환불 접수 후 판매자가 평균 48시간(영업일 2일) 안에 답변합니다."]


def _claims(*items):
    return json.dumps({"claims": [{"span": s, "claim": c, "type": t} for s, c, t in items]}, ensure_ascii=False)


def _verdicts(*items):
    return json.dumps({"verdicts": [{"i": i, "verdict": v, "source": src, "evidence": ev}
                                    for i, (v, src, ev) in enumerate(items)]}, ensure_ascii=False)


def test_extract_stage_sees_no_evidence_or_instructions():
    llm = ScriptedLLM([_claims(("판매자가 평균 2영업일 안에 답해 줘요", "판매자는 평균 2영업일 안에 답한다", "안내")),
                       _verdicts(("supported", "doc", "평균 48시간(영업일 2일) 안에 답변합니다"))])
    judge_turn(llm, say=SAY, customer="교환되나요", docs=DOCS, tools=[], instructions="지시: 상담원 연결을 제안하지 않는다")
    first = llm.calls[0][0]["content"]
    assert SAY in first and "교환되나요" in first
    assert "48시간" not in first and "상담원 연결을 제안하지" not in first   # 지시문을 바꿔도 주장 목록이 같다


def test_span_not_in_utterance_is_dropped():
    llm = ScriptedLLM([_claims(("판매자가 평균 2영업일 안에 답해 줘요", "a", "안내"), ("환불은 3일 걸려요", "b", "안내"))])
    assert [c["span"] for c in extract_claims(llm, SAY)] == ["판매자가 평균 2영업일 안에 답해 줘요"]


def test_no_claims_skips_verify_and_passes():
    llm = ScriptedLLM([_claims()])
    r = judge_turn(llm, say="사진 첨부해서 접수해 드릴까요?", customer="", docs=DOCS, tools=[], instructions="")
    assert r == {"unsupported": False, "claim": "", "claims": []} and len(llm.calls) == 1


def test_supported_needs_real_quote_and_tool_needs_success():
    llm = ScriptedLLM([
        _claims(("판매자가 평균 2영업일 안에 답해 줘요", "판매자 평균 2영업일", "안내"),
                ("사진 첨부해서", "사진 첨부가 필요하다", "안내"),
                ("답해 줘요", "링크를 보냈다", "결과")),
        _verdicts(("supported", "doc", "판매자가 평균 48시간(영업일 2일) 안에 답변합니다"),   # 원문 인용 → 인정
                  ("supported", "doc", "사진은 반드시 3장 이상 첨부"),                     # 지어낸 인용 → unverified
                  ("supported", "tool:send_app_link", "")),                             # 성공 결과 없음 → unverified
    ])
    r = judge_turn(llm, say=SAY, customer="", docs=DOCS, tools=[{"name": "send_app_link", "ok": False}], instructions="")
    assert [c["verdict"] for c in r["claims"]] == ["supported", "unverified", "unverified"]
    assert r["unsupported"] is True and "사진 첨부해서" in r["claim"]


def test_tool_result_supports_result_report():
    llm = ScriptedLLM([_claims(("답해 줘요", "링크를 보냈다", "결과")), _verdicts(("supported", "tool:send_app_link", ""))])
    r = judge_turn(llm, say=SAY, customer="", docs=[], tools=[{"name": "send_app_link", "ok": True}], instructions="")
    assert r["unsupported"] is False


def test_parse_failure_is_none():
    assert judge_turn(ScriptedLLM(["not json"]), say=SAY, customer="", docs=[], tools=[], instructions="")["unsupported"] is None


def test_quoted_in_tolerates_spacing_and_slight_trim():
    assert quoted_in("평균 48시간 (영업일 2일)", DOCS[0])
    assert quoted_in("판매자가 평균 48시간 영업일 2일 안에 답변", DOCS[0])
    assert not quoted_in("3장", DOCS[0]) and not quoted_in("판매자가 평균 7일 안에 환불을 처리합니다", DOCS[0])


def test_failed_tool_backs_failure_report_when_quoted():
    say = "분실신고 처리 중 오류가 발생했어요."
    tools = [{"name": "report_card_lost", "ok": False, "error": "backend unavailable"}]
    llm = ScriptedLLM([_claims(("분실신고 처리 중 오류가 발생했어요", "분실신고가 실패했다", "결과")),
                       _verdicts(("supported", "tool:report_card_lost", '"ok": false, "error": "backend unavailable"'))])
    assert judge_turn(llm, say=say, customer="", docs=[], tools=tools, instructions="")["unsupported"] is False


def test_elided_quote_checks_each_piece():
    doc = "교환·환불 접수 후 판매자가 평균 48시간(영업일 2일) 안에 답변합니다. 답이 없으면 자동 승인돼요."
    llm = ScriptedLLM([_claims(("판매자가 평균 2영업일 안에 답해 줘요", "a", "안내")),
                       _verdicts(("supported", "doc", "판매자가 평균 48시간 ... 답이 없으면 자동 승인돼요"))])
    assert judge_turn(llm, say=SAY, customer="", docs=[doc], tools=[], instructions="")["unsupported"] is False
    llm = ScriptedLLM([_claims(("판매자가 평균 2영업일 안에 답해 줘요", "a", "안내")),
                       _verdicts(("supported", "doc", "판매자가 평균 48시간 ... 답이 없으면 전액 환불돼요"))])
    assert judge_turn(llm, say=SAY, customer="", docs=[doc], tools=[], instructions="")["unsupported"] is True
