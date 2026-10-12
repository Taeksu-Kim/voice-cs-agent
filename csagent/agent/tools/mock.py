"""가짜 백엔드 도구. 실제 시스템 대신 CallContext.customer(시나리오 YAML의 고객 데이터)를 읽고 쓴다.

시나리오에서 조정할 수 있는 것
- customer.auth_result: "success" | "fail"           앱 인증 결과
- customer.tool_failures: {"report_card_lost": "timeout"}   도구 장애 주입 (메모 H8)
"""

from __future__ import annotations

import json
from typing import Any

from csagent.agent.state import CallState
from csagent.agent.tools.registry import ToolRegistry, ToolSpec


def _fail_if_injected(state: CallState, name: str) -> None:
    failure = state.ctx.customer.get("tool_failures", {}).get(name)
    if failure:
        raise TimeoutError(f"{name} 백엔드 응답 없음 ({failure})")


def _customer(state: CallState, key: str, default: Any = None) -> Any:
    return state.ctx.customer.get(key, default if default is not None else [])


# ---------------------------------------------------------------- 조회 (read)

def lookup_orders(state: CallState) -> dict:
    _fail_if_injected(state, "lookup_orders")
    return {"today": state.ctx.today, "orders": _customer(state, "orders")}


def lookup_transfers(state: CallState) -> dict:
    _fail_if_injected(state, "lookup_transfers")
    return {"today": state.ctx.today, "transfers": _customer(state, "transfers")}


def lookup_cards(state: CallState) -> dict:
    _fail_if_injected(state, "lookup_cards")
    return {"cards": [{"card_id": c["card_id"], "name": c["name"], "last4": c.get("last4", ""),
                       "status": c.get("status", "정상")} for c in _customer(state, "cards")]}


def lookup_seller(state: CallState) -> dict:
    _fail_if_injected(state, "lookup_seller")
    return {"seller": _customer(state, "seller", {})}


# ---------------------------------------------------------------- 확인·인증 (notify)

def request_app_auth(state: CallState) -> dict:
    _fail_if_injected(state, "request_app_auth")
    if state.ctx.customer.get("auth_result", "success") == "success":
        state.authenticated = True
        return {"status": "success"}
    state.auth_failures += 1
    return {"status": "failed", "failures": state.auth_failures}


def request_confirmation(state: CallState, action: str, target) -> dict:
    """실행 전 고객 재확인을 건다. 고객의 답은 다음 턴에 LLM이 판정하고 코드가 인용을 대조해 확정한다."""
    target = _as_target(target)
    if target is None:
        return {"error": "target은 객체여야 함 (예: {\"card_id\": \"C1\"})"}
    if state.is_confirmed(action, target):
        return {"status": "already_confirmed", "note": "이미 고객이 동의했습니다. 다시 묻지 말고 실행하세요."}
    p = state.pending
    if p and p.kind == "confirm" and p.action == action and p.target == target:
        # 같은 확인을 다시 걸면 원래 질문 턴을 유지한다 (다음 턴 판정이 유효하도록)
        return {"status": "pending", "action": action, "target": target, "note": "이미 같은 확인을 걸어 두었습니다."}
    state.ask("confirm", action, target)
    return {"status": "pending", "action": action, "target": target,
            "note": "고객에게 실행 내용을 말하고 진행 여부를 물어보세요. 답은 다음 턴에 확인됩니다."}


def _as_target(target):
    """LLM이 대상 객체를 JSON 문자열로 넣는 경우가 있다 → 객체로 해석 (형식 변환, 의미 판단 아님)."""
    if isinstance(target, dict):
        return target
    if isinstance(target, str):
        try:
            value = json.loads(target)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def ask_contact(state: CallState) -> dict:
    """아웃바운드: 통화 상대가 담당자인지 묻는다. 확정은 다음 턴 고객 발화 판정으로."""
    if state.contact_verified:
        return {"status": "already_verified", "note": "이미 담당자로 확인됐습니다. 다시 묻지 마세요."}
    state.ask("contact")
    return {"status": "pending", "note": "담당자 본인인지 물어보세요."}


# ---------------------------------------------------------------- 실행 (execute)

def _card(state: CallState, card_id: str) -> dict:
    return next((c for c in _customer(state, "cards") if c["card_id"] == card_id), None)


def suspend_card(state: CallState, card_id: str) -> dict:
    _fail_if_injected(state, "suspend_card")
    card = _card(state, card_id)
    if card is None:
        return {"error": f"unknown_card:{card_id}"}
    card["status"] = "일시정지"
    return {"status": "suspended", "card_id": card_id}


def report_card_lost(state: CallState, card_id: str) -> dict:
    _fail_if_injected(state, "report_card_lost")
    card = _card(state, card_id)
    if card is None:
        return {"error": f"unknown_card:{card_id}"}
    card["status"] = "분실신고"
    return {"status": "reported_lost", "card_id": card_id}


# ---------------------------------------------------------------- 안내·기록 (notify)

def send_app_link(state: CallState, target: str) -> dict:
    _fail_if_injected(state, "send_app_link")
    return {"sent": True, "target": target}


def schedule_callback(state: CallState, when: str) -> dict:
    return {"scheduled": True, "when": when}


def record_lead(state: CallState, need: str = "", budget: str = "", interest: str = "",
                grade: str = "") -> dict:
    return {"recorded": True, "lead": {"need": need, "budget": budget, "interest": interest, "grade": grade}}


# ---------------------------------------------------------------- 이관·종료 (handoff)

def transfer_to_agent(state: CallState, reason: str = "", summary: str = "") -> dict:
    return {"transferred": True, "reason": reason, "summary": summary}


DISPOSITIONS = ("resolved", "link_sent", "transferred", "refused", "unresolved")


def end_call(state: CallState, disposition: str = "resolved") -> dict:
    if disposition not in DISPOSITIONS:
        return {"error": f"bad_disposition:{disposition} (허용: {', '.join(DISPOSITIONS)})"}
    return {"ended": True, "disposition": disposition}


def default_registry() -> ToolRegistry:
    return ToolRegistry([
        ToolSpec("lookup_orders", "통화 중인 본인의 토스쇼핑 주문·배송 상태 조회 (다른 사람 주문은 조회 불가)",
                 fn=lookup_orders),
        ToolSpec("lookup_transfers", "본인의 최근 송금 내역 조회", fn=lookup_transfers),
        ToolSpec("lookup_cards", "본인 카드 목록 조회 (카드 이름·끝 4자리·상태)", fn=lookup_cards),
        ToolSpec("lookup_seller", "통화 상대 셀러의 상점 정보·페널티·광고 이용 조회", requires_contact=True,
                 fn=lookup_seller),
        ToolSpec("request_app_auth", "고객 휴대폰으로 앱 본인 인증 요청 (결과가 바로 돌아옴)", risk="notify",
                 fn=request_app_auth),
        ToolSpec("request_confirmation", "실행 도구를 쓰기 전 고객 재확인을 건다. 고객의 답은 다음 턴에 확인됨",
                 {"action": "실행할 도구 이름", "target": "실행 도구에 넣을 인자와 같은 JSON 객체 — 문자열 아님 (예: {\"card_id\": \"C1\"})"},
                 risk="notify", asks_customer=True, fn=request_confirmation),
        ToolSpec("ask_contact", "아웃바운드: 통화 상대가 담당자인지 확인 질문을 건다", risk="notify",
                 asks_customer=True, fn=ask_contact),
        ToolSpec("suspend_card", "카드 일시정지 (토스 비밀번호로 해제 가능)", {"card_id": "카드 ID"},
                 risk="execute", requires_auth=True, requires_confirmation=True, target_keys=("card_id",),
                 fn=suspend_card),
        ToolSpec("report_card_lost", "카드 분실신고 (해제하려면 실물 카드 CVC 필요)", {"card_id": "카드 ID"},
                 risk="execute", requires_auth=True, requires_confirmation=True, target_keys=("card_id",),
                 fn=report_card_lost),
        ToolSpec("send_app_link", "고객 휴대폰으로 토스 앱 화면 링크 문자 발송", {"target": "화면 이름 (예: 주문 내역)"},
                 risk="notify", fn=send_app_link),
        ToolSpec("schedule_callback", "재통화 예약", {"when": "희망 시각"}, risk="notify", fn=schedule_callback),
        ToolSpec("record_lead", "아웃바운드: 파악한 니즈·예산·관심도·등급 기록",
                 {"need": "니즈", "budget": "예산 범위", "interest": "관심도", "grade": "A|B|C"},
                 risk="notify", requires_contact=True, fn=record_lead),
        ToolSpec("transfer_to_agent", "사람 상담원 연결 (요약을 함께 넘김)", {"reason": "사유", "summary": "통화 요약"},
                 risk="handoff", fn=transfer_to_agent),
        ToolSpec("end_call", "통화 종료", {"disposition": "resolved|link_sent|transferred|refused|unresolved"},
                 risk="handoff", fn=end_call),
    ])
