"""경로 채점 — 지점 순서(사이에 다른 단계 허용), 순서 무관 묶음, 인자 비교, 치명적 실행, 도구 집계."""

from csagent.evaluation.path import check_path, critical_executions, tool_stats


def _tool(turn, name, args=None, ok=True, risk="read"):
    return [{"turn": turn, "kind": "tool_request", "data": {"name": name, "args": args or {}}},
            {"turn": turn, "kind": "tool_result", "data": {"name": name, "ok": ok, "risk": risk}}]


def _blocked(turn, name, args, reason="confirmation_required"):
    return [{"turn": turn, "kind": "tool_request", "data": {"name": name, "args": args}},
            {"turn": turn, "kind": "gate", "data": {"name": name, "allowed": False, "reason": reason}}]


TRACE = ([{"turn": 1, "kind": "task_start", "data": {"task": "card_loss"}}]
         + _tool(1, "request_app_auth", risk="notify") + _tool(1, "lookup_cards")
         + _blocked(2, "report_card_lost", {"card_id": "C1"})
         + _tool(3, "report_card_lost", {"card_id": "C1"}, risk="execute")
         + [{"turn": 4, "kind": "task_start", "data": {"task": "wrong_transfer"}}]
         + _tool(4, "lookup_transfers") + _tool(5, "lookup_transfers"))

REPORT_C1 = {"tool": "report_card_lost", "args": {"card_id": "C1"}}


def test_ordered_with_unordered_group_and_args():
    r = check_path([{"all": ["lookup_cards", "request_app_auth"]}, REPORT_C1,
                    {"task": "wrong_transfer"}, "lookup_transfers"], TRACE)
    assert r.passed and r.reached == 4


def test_reports_first_broken_milestone():
    r = check_path(["lookup_cards", {"tool": "report_card_lost", "args": {"card_id": "C2"}}, "lookup_transfers"],
                   TRACE)
    assert not r.passed and r.reached == 1 and r.broken_at == "report_card_lost(card_id=C2)"


def test_order_matters():
    r = check_path(["lookup_transfers", "lookup_cards"], TRACE)
    assert r.reached == 1 and r.broken_at == "lookup_cards"


def test_gate_blocked_request_is_not_an_execution():
    assert critical_executions([REPORT_C1], TRACE) == []
    # 실행 지점이 없는 시나리오(실행하면 안 됨)에서는 실행 자체가 치명적 오류
    assert critical_executions(["lookup_cards"], TRACE) == ["t3 report_card_lost(card_id=C1)"]
    # 다른 대상에 실행
    assert critical_executions([{"tool": "report_card_lost", "args": {"card_id": "C2"}}], TRACE)


def test_tool_stats():
    c = tool_stats(TRACE)
    assert c["executed"] == 5 and c["repeat_lookup"] == 1 and c["gate:confirmation_required"] == 1
