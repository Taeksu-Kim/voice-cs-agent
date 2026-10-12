import json

from csagent.agent.contracts import CallContext, TurnInput
from csagent.agent.guardrails.gate import ExecutionGate
from csagent.agent.interface import new_call
from csagent.agent.tools.mock import default_registry
from csagent.llm import ScriptedLLM
from tests.test_agent_evaluation import make_agent

CTX = CallContext(audience="consumer", today="2026-10-12", customer={
    "name": "이서연", "auth_result": "success",
    "cards": [{"card_id": "C1", "name": "토스유스카드", "last4": "1234"},
              {"card_id": "C2", "name": "토스뱅크 체크카드", "last4": "5678"}]})
TOOLS = default_registry()
GATE = ExecutionGate()


def _check(state, name, args):
    return GATE.check(state, TOOLS.get(name), args)


def _out(say="", tool=None, answer=None):
    return json.dumps({"pending_answer": answer, "say": say, "tool": tool}, ensure_ascii=False)


# ---------------------------------------------------------------- 게이트 (상태 대조)

def test_execute_requires_auth_then_confirmation_of_same_target():
    s = new_call(CTX)
    s.turn = 1
    assert _check(s, "report_card_lost", {"card_id": "C1"}).reason == "auth_required"
    TOOLS.run(s, "request_app_auth", {})
    assert _check(s, "report_card_lost", {"card_id": "C1"}).reason == "confirmation_required"

    TOOLS.run(s, "request_confirmation", {"action": "report_card_lost", "target": {"card_id": "C1"}})
    s.turn = 2
    assert s.resolve_pending("yes", "네 해주세요", "네 해주세요 빨리요")
    assert _check(s, "report_card_lost", {"card_id": "C1"}).allowed
    # 재확인한 대상과 다른 카드는 여전히 막힌다
    assert _check(s, "report_card_lost", {"card_id": "C2"}).reason == "confirmation_required"


def test_duplicate_execution_blocked():
    s = new_call(CTX)
    s.authenticated, s.turn = True, 1
    s.ask("confirm", "suspend_card", {"card_id": "C1"})
    s.turn = 2
    s.resolve_pending("yes", "네", "네")
    assert TOOLS.run(s, "suspend_card", {"card_id": "C1"}).ok
    assert _check(s, "suspend_card", {"card_id": "C1"}).reason == "duplicate"


def test_outbound_tools_need_contact_verification():
    s = new_call(CTX)
    assert _check(s, "record_lead", {"grade": "A"}).reason == "contact_required"
    s.turn = 1
    TOOLS.run(s, "ask_contact", {})
    s.turn = 2
    s.resolve_pending("yes", "제가 담당자예요", "네 제가 담당자예요")
    assert _check(s, "record_lead", {"grade": "A"}).allowed


def test_allowed_tools_restriction():
    s = new_call(CTX)
    d = GATE.check(s, TOOLS.get("lookup_orders"), {}, allowed_tools={"lookup_cards"})
    assert d.reason == "not_allowed"


# ---------------------------------------------------------------- 확인 판정 (LLM 판정 + 인용 대조)

def test_pending_answer_requires_quote_in_utterance_and_previous_turn():
    s = new_call(CTX)
    s.turn = 1
    s.ask("confirm", "report_card_lost", {"card_id": "C1"})
    assert not s.resolve_pending("yes", "네", "네")               # 같은 턴에 건 질문은 확정 불가
    s.turn = 2
    assert not s.resolve_pending("yes", "네 진행해 주세요", "음 잠깐만요")  # 고객이 하지 않은 말 인용
    assert not s.resolve_pending("unclear", "글쎄요", "글쎄요")
    assert s.pending is not None                                  # 확정 안 되면 질문 유지
    assert s.resolve_pending("no", "아니요", "아니요 그냥 둘게요")
    assert s.pending is None and not s.confirmations              # 거절은 기록만 지우고 확인은 안 생김


# ---------------------------------------------------------------- 도구 계층

def test_tool_failures_and_bad_args_are_results_not_exceptions():
    s = new_call(CallContext(customer={"cards": [{"card_id": "C1", "name": "카드"}],
                                       "tool_failures": {"report_card_lost": "timeout"}}))
    out = TOOLS.run(s, "report_card_lost", {"card_id": "C1"})
    assert not out.ok and "TimeoutError" in out.error and not s.executed
    assert TOOLS.run(s, "lookup_orders", {"wrong": 1}).error.startswith("bad_args")
    ok_backend = new_call(CallContext(customer={"cards": [{"card_id": "C1", "name": "카드"}]}))
    assert TOOLS.run(ok_backend, "report_card_lost", {"card_id": "C9"}).error.startswith("unknown_card")


def test_auth_failure_counts():
    s = new_call(CallContext(customer={"auth_result": "fail"}))
    assert TOOLS.run(s, "request_app_auth", {}).result["status"] == "failed"
    assert not s.authenticated and s.auth_failures == 1


# ---------------------------------------------------------------- A0 vs A1 (같은 LLM 출력)

def _rushing_llm():
    """인증·재확인 없이 바로 분실신고를 실행하려는 LLM."""
    return ScriptedLLM([
        _out("", {"name": "report_card_lost", "args": {"card_id": "C1"}}),
        _out("분실신고 처리했어요."),
    ])


def test_a0_executes_without_auth_but_a1_blocks():
    s0 = new_call(CTX)
    make_agent(_rushing_llm(), gate=False).respond(s0, TurnInput("카드 잃어버렸어요"))
    assert "reported_lost" in json.dumps([e.data for e in s0.trace.of("tool_result")])

    s1 = new_call(CTX)
    llm = _rushing_llm()
    reply = make_agent(llm, gate=True).respond(s1, TurnInput("카드 잃어버렸어요"))
    gate = s1.trace.of("gate")[0].data
    assert gate == {"name": "report_card_lost", "allowed": False, "reason": "auth_required"}
    assert not s1.trace.of("tool_result")                         # 실행되지 않음
    assert "[실행 차단]" in llm.calls[1][-1]["content"]           # 막힌 사유를 LLM에 돌려줌
    assert reply.tool_calls[0].result == {"ok": False, "blocked": "auth_required"}


def test_a1_full_card_loss_flow_over_two_turns():
    s = new_call(CTX)
    agent = make_agent(ScriptedLLM([
        # 턴 1: 인증 → 재확인 질문
        _out("", {"name": "request_app_auth", "args": {}}),
        _out("", {"name": "request_confirmation",
                  "args": {"action": "report_card_lost", "target": {"card_id": "C1"}}}),
        _out("토스유스카드를 분실신고할까요?"),
        # 턴 2: 고객 동의 판정(인용) + 실행
        _out("", {"name": "report_card_lost", "args": {"card_id": "C1"}},
             answer={"answer": "yes", "quote": "네 해주세요"}),
        _out("분실신고 완료됐어요."),
    ]), gate=True)
    agent.respond(s, TurnInput("유스카드 잃어버렸어요"))
    assert s.authenticated and s.pending is not None
    reply = agent.respond(s, TurnInput("네 해주세요"))
    assert reply.text == "분실신고 완료됐어요."
    assert [e.data["allowed"] for e in s.trace.of("gate") if e.data["name"] == "report_card_lost"] == [True]
    assert s.ctx.customer["cards"][0]["status"] == "분실신고"
    assert CTX.customer["cards"][0].get("status") is None          # 시나리오 원본은 그대로 (깊은 복사)


def test_end_call_rejects_unknown_disposition():
    s = new_call(CTX)
    assert TOOLS.run(s, "end_call", {"disposition": "normal"}).error.startswith("bad_disposition")
    assert TOOLS.run(s, "end_call", {"disposition": "refused"}).ok
