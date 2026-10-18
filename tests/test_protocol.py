import pytest

from csagent.agent.contracts import CallContext, TurnInput
from tests.local import call
from csagent.agent.protocol import load_protocol
from csagent.agent.state import Pending, TaskState
from tests.test_controller import CARDS, _controller, _out, _route


def _state():
    s = call(CallContext(customer=CARDS))
    s.active = TaskState("card_loss")
    return s


def test_card_loss_step_follows_code_known_facts():
    p, s = load_protocol("card_loss"), _state()
    assert p.current(s).id == "lookup"
    s.active.tools_done.add("lookup_cards")
    assert p.current(s).id == "auth"
    s.authenticated = True
    assert p.current(s).id == "choose"
    s.pending = Pending("confirm", s.turn, [("report_card_lost", {"card_id": "C1"})])
    assert p.current(s).id == "confirm"
    s.pending = None
    s.succeeded.append(("report_card_lost", {"card_id": "C1"}))
    assert p.current(s).id == "wrap_up"
    s.auth_failures = 2
    assert p.current(s).id == "auth_failed"


def test_general_step_and_tools():
    p, s = load_protocol("general"), _state()
    s.active = TaskState("general")
    assert p.current(s).id == "start"
    assert set(p.tools(p.current(s))) == {"lookup_orders", "send_app_link", "transfer_to_agent", "end_call"}
    s.active.tools_done.add("lookup_orders")
    assert p.current(s).id == "answer"


def test_step_change_mid_turn_is_announced_and_tools_follow_step():
    c, llm = _controller([
        _route("card_loss"),
        _out(say="확인해 볼게요.", tool={"name": "report_card_lost", "args": {"card_id": "C1"}}),  # 조회 전 실행 → 단계 밖
        _out(say="확인해 볼게요.", tool={"name": "lookup_cards", "args": {}}),
        _out(say="앱으로 인증 요청을 보낼게요.", tool={"name": "request_app_auth", "args": {}}),
        _out(say="카드를 다시 찾을 수 있을까요?"),
    ])
    s = call(CallContext(customer=CARDS))
    c.respond(s, TurnInput("유스카드 잃어버렸어요"))
    assert [e.data for e in s.trace.of("gate")][0] == {"name": "report_card_lost", "allowed": False,
                                                       "reason": "not_allowed"}
    assert [e.data["step"] for e in s.trace.of("protocol_step")] == ["lookup", "auth", "choose"]
    assert "[지금 단계: lookup]" in llm.calls[1][0]["content"]
    assert "[지금 단계: auth]" in llm.calls[3][-1]["content"]          # 조회 결과 뒤에 새 단계 안내
    assert "[지금 단계: choose]" in llm.calls[4][-1]["content"]


def test_unknown_condition_is_rejected_at_load(tmp_path):
    (tmp_path / "x.yaml").write_text("name: x\ncommon: c\nsteps:\n  - {id: a, say: s, when: {authenticatd: true}}\n",
                                     encoding="utf-8")
    with pytest.raises(ValueError, match="authenticatd"):
        load_protocol("x", tmp_path)


def test_no_cards_found_moves_to_not_found_step_without_auth():
    c, llm = _controller([
        _route("card_loss"),
        _out(say="확인해 볼게요.", tool={"name": "lookup_cards", "args": {}}),
        _out(say="고객님 명의로 조회되는 토스 카드가 없어요. 어느 회사 카드인가요?"),
    ])
    s = call(CallContext(customer={"name": "행복상회"}))
    c.respond(s, TurnInput("신용카드를 잃어버렸어요"))
    assert [e.data["step"] for e in s.trace.of("protocol_step")] == ["lookup", "not_found"]
    assert "request_app_auth" not in llm.calls[-1][0]["content"].split("이 단계에서 쓸 수 있는 도구:")[-1].split("\n")[0]
    # 카드가 있는 고객은 지금까지처럼 인증 단계로
    s2 = _state()
    s2.active.tools_done.add("lookup_cards")
    assert load_protocol("card_loss").current(s2).id == "auth"
