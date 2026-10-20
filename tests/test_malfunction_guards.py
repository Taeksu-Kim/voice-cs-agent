"""오동작 가드레일 (docs/09) — G1 시각 없는 예약 거절, G2 확인하겠다고 하고 끝내는 턴 되돌리기."""
import json

from fastapi.testclient import TestClient

from csagent.agent.baseline import BaselineAgent
from csagent.agent.contracts import CallContext, TurnInput
from csagent.agent.guardrails.gate import ExecutionGate
from csagent.agent.streaming import unkept_promise
from csagent.domain.app import create_app
from csagent.domain.store import FixtureStore
from csagent.llm import ScriptedLLM
from tests.local import TOOLS, call
from tests.test_agent_evaluation import FakeRetriever

CTX = CallContext(audience="consumer", today="2026-10-12",
                  customer={"name": "김민지", "orders": [{"order_id": "A1", "status": "배송중"}]})


def _say(text, tool=None):
    # 실제 출력과 같은 키 순서 — tool이 say 앞 (docs/09 G7: 말을 내보내기 전에 막힐 도구인지 본다)
    return json.dumps({"tool": tool, "say": text}, ensure_ascii=False)


def test_callback_without_time_is_rejected():
    store = FixtureStore()
    store.put("c1", {})
    c = TestClient(create_app(store))
    for when in ("unknown", "", " 미정 "):
        r = c.post("/notify/callback", json={"customer_id": "c1", "when": when}).json()
        assert r["ok"] is False and r["error"]["code"] == "WHEN_REQUIRED"
    ok = c.post("/notify/callback", json={"customer_id": "c1", "when": "오늘 오후 4시 이후"}).json()
    assert ok["data"] == {"scheduled": True, "when": "오늘 오후 4시 이후"}   # 시각 표현은 판단하지 않는다


def test_unkept_promise_only_when_turn_ends_on_promise_to_check_now():
    for say in ("토스뱅크 체크카드 상태 확인해 볼게요.", "확인해 드릴게요", "네, 알아봐 드릴게요."):
        assert unkept_promise(say)
    for say in ("네, 확인해 드릴게요. 최근 30일 위반 항목은 발송 기한 미준수 3점입니다.",   # 약속 뒤에 답이 있다
                "와이프분이 직접 전화 주시면 바로 확인해 드릴게요.",                    # 나중 일
                "확인해 볼게요?", "배송중이에요.", ""):
        assert not unkept_promise(say)


def _agent(llm):
    return BaselineAgent(FakeRetriever(), llm, TOOLS, gate=ExecutionGate())


def test_promise_without_tool_is_sent_back_once_then_tool_runs():
    llm = ScriptedLLM([_say("확인해 볼게요."), _say("", {"name": "lookup_orders", "args": {}}), _say("배송중이에요.")])
    state = call(CTX)
    reply = _agent(llm).respond(state, TurnInput("택배 어디쯤이에요"))
    assert [c.name for c in reply.tool_calls] == ["lookup_orders"]
    assert reply.text == "확인해 볼게요. 배송중이에요."            # 이미 한 말은 그대로, 결과가 이어진다
    assert "확인해 보겠다고 말했지만" in llm.calls[1][-1]["content"]
    assert [e.kind for e in state.trace.of("promise_nudge")] == ["promise_nudge"]


def test_promise_nudge_happens_only_once_per_turn():
    llm = ScriptedLLM([_say("확인해 볼게요."), _say("확인해 드릴게요.")])
    state = call(CTX)
    reply = _agent(llm).respond(state, TurnInput("택배 어디쯤이에요"))
    assert len(llm.calls) == 2 and not reply.tool_calls      # 두 번째는 되돌리지 않고 턴을 끝낸다
    assert len(state.trace.of("promise_nudge")) == 1


# -- G3 들은 말 = 기록, G4 결과 기록 사실 확인, G5 턴 오류 복구 --

from csagent.agent.controller import CallController
from csagent.agent.loop import TurnLoop
from csagent.agent.router import Router
from csagent.agent.workers.specs import WORKERS
from tests.local import looked_up

CARDS = {"name": "이서연", "cards": [{"card_id": "C1", "name": "토스유스카드", "last4": "1234"}], "auth_result": "success"}


def _out(**kw):
    out = {"intent": "continue", "pending_answer": None, "slots": {}, "tool": None, "say": "", "confirm": None}
    out.update(kw)   # 키 순서는 위 그대로 (tool이 say 앞)
    return json.dumps(out, ensure_ascii=False)


def _controller(outputs):
    llm = ScriptedLLM(outputs)
    loop = TurnLoop(FakeRetriever(), llm, TOOLS, gate=ExecutionGate())
    return CallController(loop, Router(llm, WORKERS), WORKERS), llm


def test_speech_of_step_that_will_be_blocked_is_not_sent():
    # 「분실신고 진행할게요」 + 실행 → 동의 확인에 걸림 → 「진행할까요?」 (실측 10/16~19 95턴, docs/09 G7)
    # tool이 say 앞이라 말을 내보내기 전에 막힐 걸 안다 → 고객은 질문만 듣는다
    c, llm = _controller([
        _out(say="토스유스카드 분실신고 진행할게요.", tool={"name": "report_card_lost", "args": {"card_id": "C1"}}),
        _out(say="진행할까요?"),
    ])
    s = call(CallContext(customer=CARDS, direction="inbound"))
    s.authenticated = True
    looked_up(s)
    from csagent.agent.state import TaskState
    s.active = TaskState("card_loss", started=True)
    s.active.tools_done.add("lookup_cards")
    reply = c.respond(s, TurnInput("잃어버렸어요 분실신고 해 주세요"))
    assert reply.text == "진행할까요?"
    drop = s.trace.of("unheard_drop")
    assert [(e.data["text"], e.data["blocked"]) for e in drop] == [("토스유스카드 분실신고 진행할게요.", "confirmation")]


def test_no_fallback_after_customer_already_heard_something():
    # 들은 말이 있는데 마지막에 「잘 못 들었어요」를 붙이던 것 (10/16 s5_b0_report_lost: 같은 말 4번 + 대체 문장)
    c, _ = _controller([_out(say="분실신고 진행할게요.", tool={"name": "report_card_lost", "args": {"card_id": "C9"}})] * 4)
    s = call(CallContext(customer=CARDS))
    from csagent.agent.state import TaskState
    s.authenticated = True
    looked_up(s)
    s.active = TaskState("card_loss", started=True)
    s.active.tools_done.add("lookup_cards")
    reply = c.respond(s, TurnInput("분실신고요"))
    assert "잘 못 들었어요" not in reply.text                       # 고객 탓이 아니다
    assert reply.text == "죄송해요, 확인하는 중에 문제가 생겼어요. 한 번만 다시 말씀해 주시겠어요?"


def test_end_call_disposition_must_match_what_happened():
    from csagent.agent.tools.builtin import end_call
    s = call(CTX)
    assert "disposition_mismatch" in end_call(s, "link_sent")["error"]
    assert "disposition_mismatch" in end_call(s, "transferred")["error"]
    s.trace.add(1, "tool_result", {"name": "send_app_link", "ok": True})
    assert end_call(s, "link_sent") == {"ended": True, "disposition": "link_sent"}
    assert end_call(s, "resolved") == {"ended": True, "disposition": "resolved"}


class _Broken:
    def chat(self, *a, **k):
        raise TimeoutError("LLM 응답 없음")

    def stream_chat(self, *a, **k):
        raise TimeoutError("LLM 응답 없음")


def test_turn_error_speaks_apology_keeps_history_then_hands_off():
    loop = TurnLoop(FakeRetriever(), _Broken(), TOOLS, gate=ExecutionGate())
    c = CallController(loop, Router(_Broken(), WORKERS), WORKERS)
    s = call(CTX)
    events = list(c.respond_stream(s, TurnInput("택배 어디쯤이에요")))
    assert "".join(e.data["text"] for e in events if e.type == "delta") == "죄송해요, 잠시 문제가 생겼어요. 한 번만 다시 말씀해 주시겠어요?"
    assert s.history[-2] == ("customer", "택배 어디쯤이에요")      # 고객 말이 이력에서 빠지지 않는다
    second = c.respond(s, TurnInput("택배요"))
    assert second.end_call and second.disposition == "transferred"   # 연속 오류면 상담원에게
    assert [e.data["handoff"] for e in s.trace.of("turn_error")] == [False, True]


def test_speech_streams_right_away_when_this_step_gives_consent():
    # 걸어 둔 확인 질문에 고객이 「네」 → 같은 스텝의 실행은 막히지 않는다 → 말을 붙잡지 않는다 (늦게 내보내면 첫 소리가 실행 뒤로 밀림)
    c, _ = _controller([
        _out(pending_answer={"answer": "yes", "quote": "네 해 주세요"},
             tool={"name": "report_card_lost", "args": {"card_id": "C1"}}, say="분실신고 접수할게요."),
        _out(say="접수됐어요."),
    ])
    s = call(CallContext(customer=CARDS))
    from csagent.agent.state import TaskState
    s.authenticated = True
    looked_up(s)
    s.active = TaskState("card_loss", started=True)
    s.active.tools_done.add("lookup_cards")
    s.ask_confirm("report_card_lost", {"card_id": "C1"})
    s.turn += 1   # 질문은 이전 턴에 걸렸다
    c.respond(s, TurnInput("네 해 주세요"))
    first = s.trace.of("spoken")[0].data
    assert first["text"] == "분실신고 접수할게요." and not first.get("late")
    assert any(e.data["name"] == "report_card_lost" and e.data["ok"] for e in s.trace.of("tool_result"))


def test_restarted_task_keeps_lookups_and_answers_without_relookup():
    # 업무가 done으로 끝난 뒤 같은 업무가 다시 시작되면 조회 기록을 이어받는다 (docs/09 G8)
    orders = {"name": "정우성", "orders": [{"order_id": "A5401", "item": "무선 청소기", "status": "배송완료"}]}
    c, _ = _controller([
        json.dumps({"tasks": ["exchange_return"], "reason": "반품"}),
        _out(tool={"name": "lookup_orders", "args": {}}, say="확인해 볼게요."),
        _out(intent="done", say="사진을 찍어 판매자 문의에 첨부해 주세요."),
        json.dumps({"tasks": ["exchange_return"], "reason": "같은 용건"}),
        _out(say="환불은 판매자 답변에 따라 정해져서 여기서 확답드리기 어려워요."),
    ])
    s = call(CallContext(customer=orders))
    c.respond(s, TurnInput("청소기가 안 켜져요"))
    reply = c.respond(s, TurnInput("환불은 확실히 되는 거죠?"))
    assert reply.text == "환불은 판매자 답변에 따라 정해져서 여기서 확답드리기 어려워요."
    assert not [e for e in s.trace.of("step_enforced") if e.turn == 2]
    assert s.trace.of("task_start")[-1].data["resumed"] is True


def test_first_lookup_hold_still_speaks_closing_of_other_tools():
    # 첫 조회 전 스텝이라도 종료 인사(end_call) 같은 다른 도구의 말은 버리지 않는다 (s3_b6 r1 3턴: 인사가 버려지고 끊김)
    c, _ = _controller([
        json.dumps({"tasks": ["exchange_return"], "reason": "반품"}),
        _out(tool={"name": "end_call", "args": {"disposition": "resolved"}}, say="네, 감사합니다. 좋은 하루 보내세요."),
    ])
    s = call(CallContext(customer={"name": "정우성", "orders": []}))
    reply = c.respond(s, TurnInput("아 됐어요 끊을게요"))
    assert reply.text == "확인해 볼게요. 네, 감사합니다. 좋은 하루 보내세요." and reply.end_call


def test_execute_timeout_is_unknown_outcome_and_blocks_retry_until_relookup():
    # 응답 없음 = 처리됐는지 모름 (docs/09 G9). 「처리되지 않았습니다」 단정·재시도로 두 번 실행을 막는다
    from csagent.agent.state import Confirmation, TaskState
    customer = {**CARDS, "tool_failures": {"report_card_lost": "timeout"}}
    c, llm = _controller([
        _out(tool={"name": "report_card_lost", "args": {"card_id": "C1"}}, say=""),
        _out(tool={"name": "report_card_lost", "args": {"card_id": "C1"}}, say=""),   # 바로 재시도 → 막힘
        _out(tool={"name": "transfer_to_agent", "args": {"reason": "분실신고 응답 없음"}},
             say="처리 여부를 지금 확인할 수 없어 상담원에게 연결해 드릴게요."),
    ])
    s = call(CallContext(customer=customer))
    s.authenticated = True
    looked_up(s)
    s.active = TaskState("card_loss", started=True)
    s.active.tools_done.add("lookup_cards")
    s.confirmations.append(Confirmation("report_card_lost", {"card_id": "C1"}, 0))
    reply = c.respond(s, TurnInput("네 분실신고 해 주세요"))
    feedback = llm.calls[1][-1]["content"]
    assert "[결과 확인 불가]" in feedback and "transfer_to_agent" in feedback   # 이 단계엔 조회 도구가 없다 → 연결 안내
    assert [e.data["reason"] for e in s.trace.of("gate") if not e.data["allowed"]] == ["outcome_unknown"]
    assert [e.data["name"] for e in s.trace.of("execution_unknown")] == ["report_card_lost"]
    assert reply.disposition == "transferred"


def test_successful_lookup_clears_unknown_outcome():
    s = call(CallContext(customer=CARDS))
    s.unknown_executions.add("report_card_lost:[('card_id', 'C1')]")
    from csagent.agent.state import TaskState
    s.active = TaskState("card_loss", started=True)
    loop = TurnLoop(FakeRetriever(), ScriptedLLM([]), TOOLS, gate=ExecutionGate())
    from csagent.agent.loop import TurnPolicy
    loop._run_tool(s, {"name": "lookup_cards", "args": {}}, TurnPolicy("t", "t"), {})
    assert s.unknown_executions == set()
