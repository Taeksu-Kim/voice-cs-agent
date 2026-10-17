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

    assert s.ask_confirm("report_card_lost", {"card_id": "C1"}) == "asked"
    s.turn = 2
    assert s.ask_confirm("report_card_lost", {"card_id": "C1"}) == "waiting"   # 다시 걸면 원래 질문 유지
    assert s.resolve_pending("yes", "네 해주세요", "네 해주세요 빨리요")
    assert _check(s, "report_card_lost", {"card_id": "C1"}).allowed
    # 재확인한 대상과 다른 카드는 여전히 막힌다
    assert _check(s, "report_card_lost", {"card_id": "C2"}).reason == "confirmation_required"


def test_duplicate_execution_blocked():
    s = new_call(CTX)
    s.authenticated, s.turn = True, 1
    s.ask_confirm("suspend_card", {"card_id": "C1"})
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
    s.ask_confirm("report_card_lost", {"card_id": "C1"})
    assert not s.resolve_pending("yes", "네", "네")               # 같은 턴에 건 질문은 확정 불가
    s.turn = 2
    assert not s.resolve_pending("yes", "네 진행해 주세요", "음 잠깐만요")  # 고객이 하지 않은 말 인용
    assert not s.resolve_pending("unclear", "글쎄요", "글쎄요")
    assert s.pending is not None                                  # 확정 안 되면 질문 유지
    assert s.resolve_pending("no", "아니요", "아니요 그냥 둘게요")
    assert s.pending is None and not s.confirmations              # 거절은 기록만 지우고 확인은 안 생김


# ---------------------------------------------------------------- 도구 계층

def test_lookup_orders_by_spoken_number_only_returns_own_registered_order():
    s = new_call(CallContext(customer={"orders": [{"order_id": "A3001"}, {"order_id": "A3002"}]}))
    assert len(TOOLS.run(s, "lookup_orders", {}).result["orders"]) == 2
    assert len(TOOLS.run(s, "lookup_orders", {"order_id": ""}).result["orders"]) == 2
    assert TOOLS.run(s, "lookup_orders", {"order_id": " a3002 "}).result["orders"] == [{"order_id": "A3002"}]
    miss = TOOLS.run(s, "lookup_orders", {"order_id": "1004"})
    assert miss.ok and miss.result["orders"] == [] and miss.result["not_found"] == "1004"


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
        # 턴 1: 인증 → 실행 요청 → 동의가 없어 코드가 확인 질문을 건다
        _out("", {"name": "request_app_auth", "args": {}}),
        _out("", {"name": "report_card_lost", "args": {"card_id": "C1"}}),
        _out("토스유스카드를 분실신고할까요?"),
        # 턴 2: 고객 동의 판정(인용) + 실행
        _out("", {"name": "report_card_lost", "args": {"card_id": "C1"}},
             answer={"answer": "yes", "quote": "네 해주세요"}),
        _out("분실신고 완료됐어요."),
    ]), gate=True)
    agent.respond(s, TurnInput("유스카드 잃어버렸어요"))
    assert s.authenticated and s.pending.items == [("report_card_lost", {"card_id": "C1"})]
    reply = agent.respond(s, TurnInput("네 해주세요"))
    assert reply.text == "분실신고 완료됐어요."
    assert [e.data["allowed"] for e in s.trace.of("gate") if e.data["name"] == "report_card_lost"] == [False, True]
    assert not audit(s.trace, TOOLS).violations
    assert s.ctx.customer["cards"][0]["status"] == "분실신고"
    assert CTX.customer["cards"][0].get("status") is None          # 시나리오 원본은 그대로 (깊은 복사)


def test_end_call_rejects_unknown_disposition():
    s = new_call(CTX)
    assert TOOLS.run(s, "end_call", {"disposition": "normal"}).error.startswith("bad_value:disposition")
    assert TOOLS.run(s, "end_call", {"disposition": "refused"}).ok


# ---------------------------------------------------------------- 평가: 절차 감사·제때 조회·반복

from csagent.evaluation.procedure import audit  # noqa: E402
from csagent.evaluation.runner import run_scenario  # noqa: E402
from csagent.evaluation.scenario import Expectation, Scenario  # noqa: E402
from csagent.evaluation.scoring import score  # noqa: E402
from csagent.evaluation.simulator import ScriptedCustomer  # noqa: E402


def _rushing_llm_twice():
    return ScriptedLLM([_out("", {"name": "report_card_lost", "args": {"card_id": "C1"}}),
                        _out("분실신고 처리했어요.")])


def test_audit_counts_violation_for_a0_and_block_for_a1():
    s0 = new_call(CTX)
    make_agent(_rushing_llm_twice(), gate=False).respond(s0, TurnInput("카드 잃어버렸어요"))
    r0 = audit(s0.trace, TOOLS)
    assert len(r0.violations) == 1 and r0.violations[0]["missing"] == ["auth", "confirmation"]

    s1 = new_call(CTX)
    make_agent(_rushing_llm_twice(), gate=True).respond(s1, TurnInput("카드 잃어버렸어요"))
    r1 = audit(s1.trace, TOOLS)
    assert not r1.violations and r1.gate_blocks[0]["reason"] == "auth_required"
    assert r1.failed_exec_turns == {1}      # 이 턴은 「실패 후 완료 발언」 Judge 판정 대상


def test_required_in_time_and_no_repeat():
    sc = Scenario("t", "S1", "B0", CTX, {"goal": "택배"},
                  Expectation(tools_required=["lookup_orders"], required_by_turn=1), max_turns=3)
    llm = ScriptedLLM([_out("기다려 주세요."), _out("", {"name": "lookup_orders", "args": {}}),
                       _out("기다려 주세요.")])
    tr = run_scenario(make_agent(llm), ScriptedCustomer(["제 택배요", "언제 와요"]), sc)
    checks = {c.name: c.passed for c in score(sc, tr, TOOLS).checks}
    assert checks["tools_required"] is True       # 결국 조회는 했지만
    assert checks["required_in_time"] is False    # 1턴(답하기 전)에는 안 했음
    assert checks["no_repeat"] is False           # 같은 응답 반복


def test_same_answer_to_same_question_is_not_a_repeat():
    sc = Scenario("t", "S1", "B0", CTX, {"goal": "택배"}, Expectation(), max_turns=3)
    llm = ScriptedLLM([_out("송장이 먼저 등록된 경우일 가능성이 커요."), _out("송장이 먼저 등록된 경우일 가능성이 커요.")])
    tr = run_scenario(make_agent(llm), ScriptedCustomer(["왜 조회가 안 돼요?", "왜 조회가 안 돼요?"]), sc)
    assert {c.name: c.passed for c in score(sc, tr, TOOLS).checks}["no_repeat"] is True


def test_one_question_confirms_several_targets():
    s = new_call(CTX)
    s.authenticated, s.turn = True, 1
    s.ask_confirm("report_card_lost", {"card_id": "C1"})
    s.ask_confirm("report_card_lost", {"card_id": "C2"})            # 같은 턴 → 한 질문으로 묶임
    s.turn = 2
    assert s.resolve_pending("yes", "두 장 다 해주세요", "네 두 장 다 해주세요")
    assert _check(s, "report_card_lost", {"card_id": "C1"}).allowed
    assert _check(s, "report_card_lost", {"card_id": "C2"}).allowed


def test_confirmation_matches_by_execution_target_keys():
    from csagent.agent.state import covers
    assert covers({"card_id": "C1", "name": "토스유스카드"}, {"card_id": "C1"})
    assert not covers({"card_id": "C2", "name": "토스유스카드"}, {"card_id": "C1"})
    assert not covers({"name": "토스유스카드"}, {"card_id": None})


def test_end_call_resolved_blocked_until_confirmed_execution_succeeds():
    s = new_call(CallContext(audience="consumer", customer={**CTX.customer,
                                                                         "tool_failures": {"report_card_lost": "timeout"}}))
    s.authenticated, s.turn = True, 1
    s.ask_confirm("report_card_lost", {"card_id": "C1", "name": "토스유스카드"})
    s.turn = 2
    assert s.resolve_pending("yes", "네", "네 해주세요")
    assert _check(s, "end_call", {"disposition": "resolved"}).reason == "confirmed_not_executed"
    assert not TOOLS.run(s, "report_card_lost", {"card_id": "C1"}).ok      # 실행은 시도했지만 백엔드 장애
    assert _check(s, "end_call", {"disposition": "resolved"}).reason == "confirmed_not_executed"  # 실패도 해결 아님
    assert _check(s, "transfer_to_agent", {"reason": "장애", "summary": "-"}).allowed


def test_stall_note_added_after_two_near_identical_replies():
    from csagent.agent.loop import TurnLoop
    s = new_call(CTX)
    s.history = [("customer", "a"), ("agent", "생년월일로는 확인할 수 없어요."),
                 ("customer", "b"), ("agent", "생년월일로는 확인할 수 없어요!")]
    assert TurnLoop._stalled(s)
    s.history[-1] = ("agent", "상담원에게 연결해 드릴까요?")
    assert not TurnLoop._stalled(s)


def test_wrong_card_id_is_returned_before_asking_customer():
    s = new_call(CTX)
    s.authenticated = True
    agent = make_agent(ScriptedLLM([
        _out("", {"name": "report_card_lost", "args": {"card_id": "토스유스카드 1234"}}),   # 이름을 넣음
        _out("", {"name": "report_card_lost", "args": {"card_id": "C1"}}),
        _out("토스유스카드를 분실신고할까요?"),
    ]), gate=True)
    agent.respond(s, TurnInput("유스카드 분실신고 해주세요"))
    assert s.trace.of("precheck")[0].data["name"] == "report_card_lost"
    assert s.pending.items == [("report_card_lost", {"card_id": "C1"})]     # 틀린 값으로는 묻지 않았다
