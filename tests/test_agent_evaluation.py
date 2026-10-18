import json
from pathlib import Path

from csagent.agent.baseline import BaselineAgent
from csagent.agent.contracts import CallContext, TurnInput
from csagent.agent.guardrails.gate import ExecutionGate
from tests.local import TOOLS, call, fixture, run_call
from csagent.evaluation.scenario import Expectation, Scenario, load_scenarios
from csagent.evaluation.procedure import audit
from csagent.evaluation.scoring import LLMJudge, score
from csagent.evaluation.simulator import ScriptedCustomer
from csagent.llm import ScriptedLLM, parse_json
from csagent.retrieval.models import Hit


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, k=5, *, flt=None, **_):
        self.calls.append((query, flt))
        return [Hit("faq:4705#0", 1.0, "faq:4705", "faq", "배송 기간", "", "평균 3-5일\n[앱 링크: 주문 내역]")]


def _say(text, tool=None):
    return json.dumps({"say": text, "tool": tool}, ensure_ascii=False)


CTX = CallContext(audience="consumer", today="2026-10-12",
                  customer={"name": "김민지", "orders": [{"order_id": "A1", "status": "배송중"}]})


def make_agent(llm, retriever=None, gate=True):
    return BaselineAgent(retriever or FakeRetriever(), llm, TOOLS,
                         gate=ExecutionGate() if gate else None)


def ask(agent, text, state=None):
    state = state or call(CTX)
    return agent.respond(state, TurnInput(text)), state


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"say": "네"}\n```') == {"say": "네"}


def test_agent_lookup_tool_then_answer():
    llm = ScriptedLLM([_say("", {"name": "lookup_orders", "args": {}}), _say("배송중이에요.")])
    retriever = FakeRetriever()
    reply, _ = ask(make_agent(llm, retriever), "내 택배 어디쯤이에요")
    assert reply.text == "배송중이에요."
    assert [c.name for c in reply.tool_calls] == ["lookup_orders"]
    assert reply.tool_calls[0].result["ok"] and reply.tool_calls[0].result["orders"][0]["order_id"] == "A1"
    assert "[도구 결과]" in llm.calls[1][-1]["content"]          # 조회 결과를 다시 넣고 답했다
    assert retriever.calls[0][1].audience == "consumer"         # 통화 맥락으로 필터


def test_agent_action_tool_sets_disposition():
    llm = ScriptedLLM([_say("상담원 연결해 드릴게요.", {"name": "transfer_to_agent", "args": {"reason": "지연"}})])
    reply, _ = ask(make_agent(llm), "상담원 바꿔주세요")
    assert reply.end_call and reply.disposition == "transferred"


def test_agent_falls_back_on_bad_json():
    reply, _ = ask(make_agent(ScriptedLLM(["말도 안 되는 출력"])), "어")
    assert "다시 말씀" in reply.text


def _scenario(**expect):
    return Scenario("t1", "S1", "B0", CTX, {"goal": "배송 기간"}, Expectation(**expect), max_turns=4)


def test_runner_and_rule_scoring():
    llm = ScriptedLLM([_say("평균 삼에서 오 영업일 걸려요."),
                       _say("감사합니다. 좋은 하루 보내세요.", {"name": "end_call", "args": {"disposition": "resolved"}})])
    tr = run_call(make_agent(llm), ScriptedCustomer(["배송 며칠 걸려요", "네 감사합니다"]),
                      _scenario(dispositions=["resolved"], tools_forbidden=["lookup_orders"],
                                gold_docs=["faq:4705"]))
    assert tr.disposition == "resolved" and tr.end_reason == "agent_ended"
    sco = score(_scenario(dispositions=["resolved"], tools_forbidden=["lookup_orders"],
                          gold_docs=["faq:4705"]), tr, TOOLS)
    assert sco.passed, [c.__dict__ for c in sco.failed()]


def test_voice_format_check_flags_lists_and_parens():
    llm = ScriptedLLM([_say("고객센터(1644-2944)로\n- 문의하세요")])
    tr = run_call(make_agent(llm), ScriptedCustomer(["지연돼요"]), _scenario())
    names = {c.name for c in score(_scenario(), tr, TOOLS).failed()}
    assert "voice_format" in names


def test_customer_hangup_is_neutral_and_disposition_not_scored():
    llm = ScriptedLLM([_say("확인해 볼게요.")])
    sc = _scenario(dispositions=["resolved"])
    tr = run_call(make_agent(llm), ScriptedCustomer(["택배요"], satisfied=False), sc)
    assert tr.disposition == "customer_ended" and tr.customer_satisfied is False
    disp = next(c for c in score(sc, tr, TOOLS).checks if c.name == "disposition")
    assert disp.passed is None   # 고객이 끊은 통화의 해결 여부는 Judge(goal_resolved)가 판정


def test_judge_yes_no_to_pass_fail_and_goal_and_grounded():
    sc = _scenario(must_mention=["평균 3~5일"], must_not=["도착일 확정"])
    llm = ScriptedLLM([_say("평균 삼에서 오 일이요.")])
    tr = run_call(make_agent(llm), ScriptedCustomer(["며칠요"], satisfied=True), sc)
    judge = LLMJudge(ScriptedLLM([
        json.dumps({"answers": [
            {"id": "mention_0", "answer": "YES", "reason": "말함"},
            {"id": "must_not_0", "answer": "NO", "reason": "안 함"},
            {"id": "goal_resolved", "answer": "YES", "reason": "해결"}]}),
        json.dumps({"unsupported": "NO", "claim": ""}),   # 1턴 근거 판정
    ]))
    checks = {c.name: c.passed for c in judge.checks(sc, tr, audit(tr.state.trace, TOOLS))}
    assert checks == {"mention_0": True, "must_not_0": True, "goal_resolved": True, "grounded": True}


def test_scenario_files_load():
    scs = load_scenarios(Path(__file__).resolve().parent.parent / "scenarios")
    assert len(scs) >= 15 and {s.task for s in scs} >= {"S1", "S4", "S5", "O1"}


def test_say_extractor_streams_only_say_value():
    from csagent.agent.streaming import SayExtractor
    raw = '{"say": "배송은 \\"평균\\" 3일\\n이에요 \\uAC00", "tool": null}'
    ex, out = SayExtractor(), ""
    for i in range(0, len(raw), 2):  # 2글자씩 쪼개 들어와도
        out += ex.feed(raw[i:i + 2])
    assert out == '배송은 "평균" 3일\n이에요 가' and ex.done


def test_announced_notify_ends_turn_with_short_done_line():
    from csagent.agent.streaming import announced_done
    assert announced_done("send_app_link", "확인해 볼게요.") is None        # 미리 알리지 않았으면 결과 스텝이 말한다
    assert announced_done("request_app_auth", "인증 요청을 보낼게요.") is None   # 결과를 말해야 하는 도구는 대상 아님
    llm = ScriptedLLM([_say("", {"name": "lookup_orders", "args": {}}),
                       _say("앱 링크 보내 드릴게요.", {"name": "send_app_link", "args": {"target": "주문 내역"}}),
                       _say("보내 드렸습니다. 링크에서 확인해 주세요.")])   # 불리면 안 되는 결과 스텝
    reply = make_agent(llm).respond(call(CTX), TurnInput("주문 내역 링크 보내 주세요"))
    assert reply.text == "앱 링크 보내 드릴게요. 보내 드렸어요."


def test_respond_stream_event_order():
    llm = ScriptedLLM([_say("", {"name": "lookup_orders", "args": {}}), _say("배송중이에요.")])
    events = list(make_agent(llm).respond_stream(call(CTX), TurnInput("택배요")))
    types = [e.type for e in events]
    assert types[0] == "retrieval" and types[-1] == "done"
    assert types.index("tool_call") < types.index("tool_result") < types.index("delta")
    assert "".join(e.data["text"] for e in events if e.type == "delta") == "배송중이에요."


def test_judge_uses_transfer_goal_and_worker_instructions_and_sim_diagnostic():
    sc = _scenario(must_mention=[], must_not=[])
    tr = run_call(make_agent(ScriptedLLM([_say("확인해 볼게요.")])),
                      ScriptedCustomer(["며칠요"]), sc)
    tr.disposition = "transferred"   # 기대 결과를 정하지 않은 시나리오 → 연결이 허용된 결과
    judge_llm = ScriptedLLM([
        json.dumps({"answers": [{"id": "goal_resolved", "answer": "YES", "reason": "연결 사유 안내"}]}),
        json.dumps({"unsupported": "NO", "claim": ""}),
        json.dumps({"invented": "YES", "what": "1. 이름 김철수"}),
    ])
    judge = LLMJudge(judge_llm, {"baseline": "지시문: 토스 비밀번호로 해제"})
    score_ = score(sc, tr, TOOLS, judge)
    goal_prompt, grounded_prompt, sim_prompt = (c[0]["content"] for c in judge_llm.calls)
    assert "사람 상담원에게 연결됐다" in goal_prompt
    assert "토스 비밀번호로 해제" in grounded_prompt          # 그 턴 워커의 지시문도 근거
    assert "며칠요" in sim_prompt
    assert score_.passed                                         # 진단은 통과 판정에 안 들어감
    assert [(c.name, c.passed) for c in score_.diagnostics if c.name == "sim_faithful"] == [("sim_faithful", False)]


def test_speech_diagnostics_counts_questions_long_utterances_and_unsaid_speech():
    from csagent.evaluation.scoring import speech_diagnostics
    replies = ["카드 확인했어요. 두 장 맞죠? 분실신고할까요?", "7점이고 3점, 2점, 2점이에요."]
    trace = [{"turn": 1, "kind": "spoken", "data": {"text": "카드 확인했어요."}},
             {"turn": 1, "kind": "spoken", "data": {"text": "처리해 드릴게요."}},      # 차단돼 버려진 말
             {"turn": 2, "kind": "spoken", "data": {"text": "가" * 121}}]
    d = {c.name: c for c in speech_diagnostics(replies, trace)}
    assert d["questions_per_turn"].passed is False and "A1" in d["questions_per_turn"].detail
    assert d["numbers_per_turn"].detail.startswith("최대 4개 (A2)")
    assert d["utterance_len"].passed is False
    assert d["spoken_unsaid"].passed is False and "처리해 드릴게요" in d["spoken_unsaid"].detail
    old = {c.name: c for c in speech_diagnostics(replies, [])}   # spoken 기록 없는 이전 실행
    assert old["utterance_len"].passed is None and old["spoken_unsaid"].passed is None


def test_llm_customer_enforces_schema_and_raises_on_missing_utterance():
    import pytest
    from csagent.evaluation.simulator import SCHEMA, LLMCustomer
    sc = _scenario()
    llm = ScriptedLLM([json.dumps({"utteration": "네 해주세요", "hang_up": False, "satisfied": False})])
    with pytest.raises(ValueError):            # 키 오타 → 끊음이 아니라 오류
        LLMCustomer(llm).next(sc, [("customer", "여보세요"), ("agent", "무엇을 도와드릴까요?")])
    ok = ScriptedLLM([json.dumps({"utterance": "", "hang_up": True, "satisfied": True})])
    cust = LLMCustomer(ok)
    assert cust.next(sc, [("customer", "여보세요"), ("agent", "감사합니다.")]) is None   # 말없이 끊는 건 정상
    assert cust.satisfied is True
    assert SCHEMA["required"] == ["utterance", "hang_up", "satisfied"]
