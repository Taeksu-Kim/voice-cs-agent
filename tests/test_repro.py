"""재현기 (csagent/evaluation/repro.py) — 상태 되살리기·실패 술어·턴 재현."""
import json

import pytest

from csagent.agent.baseline import BaselineAgent
from csagent.agent.contracts import CallContext, TurnInput
from csagent.agent.guardrails.gate import ExecutionGate
from csagent.agent.state import CallState
from csagent.evaluation.repro import ReproError, ReproSpec, judge_preds, restore, run_turn
from csagent.evaluation.runner import Transcript
from csagent.evaluation.scenario import Expectation, Scenario
from csagent.llm import ScriptedLLM
from tests.local import STACK, TOOLS
from tests.test_agent_evaluation import FakeRetriever

CTX = CallContext(audience="consumer", today="2026-10-12",
                  customer={"name": "김민지", "orders": [{"order_id": "A1", "status": "배송중"}]})
SC = Scenario(id="t_repro", task="S1", behavior="-", context=CTX, customer={"goal": "-"}, expect=Expectation())


def _say(text, tool=None):
    return json.dumps({"tool": tool, "say": text}, ensure_ascii=False)


def _agent(lines):
    return BaselineAgent(FakeRetriever(), ScriptedLLM(lines), TOOLS, gate=ExecutionGate())


def _recorded_call():
    """2턴짜리 기록 (평가 results.jsonl 한 줄 모양)."""
    agent, state = _agent([_say("", {"name": "lookup_orders", "args": {}}), _say("배송중이에요."),
                           _say("네, 내일 도착 예정이에요.")]), STACK.open(SC.context)
    replies = [agent.respond(state, TurnInput("택배 어디쯤이에요")), agent.respond(state, TurnInput("언제 와요"))]
    STACK.close(state)
    return {"scenario_id": SC.id, "turns": state.history, "trace": state.trace.to_dicts(),
            "replies": [{"text": r.text, "tools": [(c.name, c.args, c.result) for c in r.tool_calls]} for r in replies]}


def test_snapshot_restore_round_trip():
    s = CallState(ctx=CTX)
    s.ask_confirm("report_card_lost", {"card_id": "C1"})
    s.executed.add("x")
    s2 = CallState(ctx=CTX)
    s2.restore(json.loads(json.dumps(s.snapshot())))
    assert s2.snapshot() == s.snapshot()


def test_restore_rebuilds_state_at_turn_start():
    row = _recorded_call()
    state, utterance, _ = restore(STACK, SC, row, 2)
    try:
        assert utterance == "언제 와요" and state.turn == 1
        assert state.history[-2:] == [("customer", "택배 어디쯤이에요"), ("agent", "배송중이에요.")]
        assert all(e.turn < 2 for e in state.trace.events)
        state.trace.add(2, "turn_start", {})
        assert state.trace.events[-1].at == "t2.s0.turn_start"   # 이어 기록해도 좌표가 겹치지 않는다
    finally:
        STACK.close(state)


def test_run_turn_counts_failure_with_current_agent():
    row = _recorded_call()
    spec = ReproSpec.from_dict({"case": "x/y", "at": "t2", "fail_if": [{"len_over": 20}, {"repeat": True}]})
    long = _agent([_say("내일 도착 예정이에요. 배송 기사님이 오후에 방문하실 예정이고 문자로 안내가 갑니다.")])
    r = run_turn(long, STACK, SC, row, spec)
    assert r["failed"] and r["hits"] == ["len_over: 20"]
    same = _agent([_say("배송중이에요.")])
    assert run_turn(same, STACK, SC, row, spec)["hits"] == ["repeat: True"]   # 바로 앞 상담원 말과 같다
    ok = _agent([_say("내일 와요.")])
    assert not run_turn(ok, STACK, SC, row, spec)["failed"]


def test_spec_validation():
    with pytest.raises(ReproError):
        ReproSpec.from_dict({"case": "x/y", "at": "t1", "fail_if": []})        # 실패 기준을 먼저 정해야 한다
    with pytest.raises(ReproError):
        ReproSpec.from_dict({"case": "x/y", "at": "t1", "mode": "x", "fail_if": [{"len_over": 1}]})
    assert ReproSpec.from_dict({"case": "x/y", "at": "t3.s2", "fail_if": [{"len_over": 1}]}).coordinate() == (3, 2)


def test_preds_on_tools_events_and_gate():
    from csagent.agent.contracts import AgentReply, ToolCall
    new = [AgentReply("네", [ToolCall("end_call", {})])]
    ev = [{"kind": "gate", "data": {"reason": "result_unheard"}}, {"kind": "unheard_drop", "data": {}}]
    hits = judge_preds([{"tool": "end_call"}, {"no_tool": "lookup_orders"}, {"gate": "result_unheard"},
                        {"event": "unheard_drop"}, {"regex": "아니"}],
                       new=new, history=[("customer", "a"), ("agent", "네")], events=ev)
    assert hits == ["tool: end_call", "no_tool: lookup_orders", "gate: result_unheard", "event: unheard_drop"]


def test_continue_with_recorded_customer_replays_later_lines():
    row = _recorded_call()
    spec = ReproSpec.from_dict({"case": "x/y", "at": "t1", "mode": "continue", "fail_if": [{"check": "no_repeat"}]})
    agent = _agent([_say("배송중이에요."), _say("배송중이에요.")])   # 고객이 「언제 와요」로 바꿔 물어도 같은 답
    r = run_turn(agent, STACK, SC, row, spec)
    assert r["texts"] == ["배송중이에요.", "배송중이에요."] and r["hits"] == ["check: no_repeat"]
