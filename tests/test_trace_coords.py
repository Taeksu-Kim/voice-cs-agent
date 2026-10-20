"""trace 좌표(at·step·where)·턴 시작 상태 스냅샷·mermaid 그림 (csagent/agent/trace.py, csagent/tracegraph.py)."""
import json

from csagent.agent.baseline import BaselineAgent
from csagent.agent.contracts import CallContext, TurnInput
from csagent.agent.guardrails.gate import ExecutionGate
from csagent.llm import ScriptedLLM
from csagent.tracegraph import coords, to_mermaid
from tests.local import TOOLS, call
from tests.test_agent_evaluation import FakeRetriever

CTX = CallContext(audience="consumer", today="2026-10-12",
                  customer={"name": "김민지", "orders": [{"order_id": "A1", "status": "배송중"}]})


def _say(text, tool=None):
    return json.dumps({"tool": tool, "say": text}, ensure_ascii=False)


def _run_two_turns():
    llm = ScriptedLLM([_say("", {"name": "lookup_orders", "args": {}}), _say("배송중이에요."), _say("네, 좋은 하루 되세요.")])
    agent, state = BaselineAgent(FakeRetriever(), llm, TOOLS, gate=ExecutionGate()), call(CTX)
    agent.respond(state, TurnInput("택배 어디쯤이에요"))
    agent.respond(state, TurnInput("알겠어요"))
    return state


def test_events_carry_turn_step_coordinates():
    state = _run_two_turns()
    by = {e.at: e for e in state.trace.events}
    assert by["t1.s0.turn_start"].step == 0
    assert by["t1.s1.tool_result"].data["name"] == "lookup_orders"   # 1번째 LLM 스텝의 도구
    assert by["t1.s2.llm"] and by["t2.s1.llm"]                       # 턴마다 스텝을 새로 센다
    assert len({e.at for e in state.trace.events}) == len(state.trace.events)   # 좌표는 통화 안에서 하나뿐
    assert all(e.where for e in state.trace.events if e.turn > 0 and e.kind != "turn_start" and e.kind != "state")


def test_turn_start_records_state_snapshot_and_call_open_records_fixture():
    state = _run_two_turns()
    snaps = [e for e in state.trace.events if e.kind == "state"]
    assert [e.at for e in snaps] == ["t1.s0.state", "t2.s0.state"]
    assert snaps[0].data["executed"] == [] and snaps[0].data["pending"] is None
    opened = state.trace.of("call_open")[0]
    assert opened.data["fixture"]["orders"][0]["order_id"] == "A1"
    json.dumps(state.trace.to_dicts(), ensure_ascii=False)   # 파일로 남길 수 있어야 한다 (집합은 목록으로)


def test_old_trace_without_coordinates_gets_inferred_steps():
    old = [{"turn": 1, "kind": "turn_start", "data": {"utterance": "x"}},
           {"turn": 1, "kind": "spoken", "data": {"text": "확인해 볼게요."}},
           {"turn": 1, "kind": "llm", "data": {"ms": 100}},
           {"turn": 1, "kind": "gate", "data": {"name": "report_card_lost", "allowed": False,
                                                "reason": "confirmation_required"}},
           {"turn": 1, "kind": "llm", "data": {"ms": 100}}]
    ats = [e["at"] for e in coords(old)]
    assert ats == ["t1.s0.turn_start", "t1.s1.spoken", "t1.s1.llm", "t1.s1.gate", "t1.s2.llm"]
    graph = to_mermaid(old, "제목", ["grounded: 근거 없음"])
    assert 't1_s1_gate["s1 gate ✗ confirmation_required report_card_lost"]' in graph
    assert "class t1_s1_gate guard" in graph and "실패: grounded" in graph
