"""가짜 백엔드 도구. 실제 시스템 대신 CallContext.customer(시나리오 YAML의 고객 데이터)를 읽고 쓴다.

시나리오에서 조정할 수 있는 것
- customer.auth_result: "success" | "fail"           앱 인증 결과
- customer.tool_failures: {"report_card_lost": "timeout"}   도구 장애 주입 (메모 H8)
"""

from __future__ import annotations

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

def lookup_orders(state: CallState, order_id: str = "") -> dict:
    """order_id가 있으면 그 건만. 본인 명의로 등록된 번호가 아니면 빈 목록 + not_found (다른 주문으로 대신 답하지 않게)."""
    _fail_if_injected(state, "lookup_orders")
    orders = _customer(state, "orders")
    key = "".join(str(order_id or "").split()).upper()
    if not key:
        return {"today": state.ctx.today, "orders": orders}
    hit = [o for o in orders if str(o.get("order_id", "")).upper() == key]
    return {"today": state.ctx.today, "orders": hit, **({} if hit else {"not_found": key})}


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


def ask_contact(state: CallState) -> dict:
    """아웃바운드: 통화 상대가 담당자인지 묻는다. 확정은 다음 턴 고객 발화 판정으로."""
    if state.contact_verified:
        return {"status": "already_verified", "note": "이미 담당자로 확인됐습니다. 다시 묻지 마세요."}
    state.ask("contact")
    return {"status": "pending", "note": "담당자 본인인지 물어보세요."}


# ---------------------------------------------------------------- 실행 (execute)

def _card(state: CallState, card_id: str) -> dict:
    return next((c for c in _customer(state, "cards") if c["card_id"] == card_id), None)


def _known_card(state: CallState, args: dict) -> str | None:
    """card_id가 조회된 카드 ID인가 (값 대조). 이름·끝 4자리를 넣으면 고객에게 묻기 전에 돌려보낸다."""
    ids = [c["card_id"] for c in _customer(state, "cards")]
    if args.get("card_id") not in ids:
        return f"card_id는 lookup_cards 결과의 card_id 중 하나여야 합니다 (받은 값: {args.get('card_id')}, 가능: {', '.join(ids)})"
    return None


def suspend_card(state: CallState, card_id: str) -> dict:
    _fail_if_injected(state, "suspend_card")
    card = _card(state, card_id)
    if card is None:
        return {"error": f"unknown_card:{card_id} (조회된 card_id: {', '.join(c['card_id'] for c in _customer(state, 'cards'))})"}
    card["status"] = "일시정지"
    return {"status": "suspended", "card_id": card_id}


def report_card_lost(state: CallState, card_id: str) -> dict:
    _fail_if_injected(state, "report_card_lost")
    card = _card(state, card_id)
    if card is None:
        return {"error": f"unknown_card:{card_id} (조회된 card_id: {', '.join(c['card_id'] for c in _customer(state, 'cards'))})"}
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
        ToolSpec("lookup_orders", "통화 중인 본인의 토스쇼핑 주문·배송 상태 조회 (다른 사람 주문은 조회 불가). "
                 "order_id를 주면 그 주문만, 비우면 본인 주문 전체",
                 {"order_id": "고객이 말한 주문번호 (말하지 않았으면 비움)"}, fn=lookup_orders),
        ToolSpec("lookup_transfers", "본인의 최근 송금 내역 조회", fn=lookup_transfers),
        ToolSpec("lookup_cards", "본인 카드 목록 조회 (카드 이름·끝 4자리·상태)", fn=lookup_cards),
        ToolSpec("lookup_seller", "통화 상대 셀러의 상점 정보·페널티·광고 이용 조회", requires_contact=True,
                 fn=lookup_seller),
        ToolSpec("request_app_auth", "고객 휴대폰으로 앱 본인 인증 요청 (결과가 바로 돌아옴)", risk="notify",
                 fn=request_app_auth),
        ToolSpec("ask_contact", "아웃바운드: 통화 상대가 담당자인지 확인 질문을 건다", risk="notify",
                 asks_customer=True, fn=ask_contact),
        ToolSpec("suspend_card", "카드 일시정지 (토스 비밀번호로 해제 가능)", {"card_id": "lookup_cards 결과의 card_id (예: C1)"},
                 risk="execute", requires_auth=True, requires_confirmation=True, target_keys=("card_id",),
                 precheck=_known_card,
                 fn=suspend_card),
        ToolSpec("report_card_lost", "카드 분실신고 (해제하려면 실물 카드 CVC 필요)", {"card_id": "lookup_cards 결과의 card_id (예: C1)"},
                 risk="execute", requires_auth=True, requires_confirmation=True, target_keys=("card_id",),
                 precheck=_known_card,
                 fn=report_card_lost),
        ToolSpec("send_app_link", "고객 휴대폰으로 토스 앱 화면 링크 문자 발송", {"target": "화면 이름 (예: 주문 내역)"},
                 risk="notify", fn=send_app_link),
        ToolSpec("schedule_callback", "재통화 예약", {"when": "희망 시각"}, risk="notify", fn=schedule_callback),
        ToolSpec("record_lead", "아웃바운드: 파악한 니즈·예산·관심도·등급 기록",
                 {"need": "니즈", "budget": "예산 범위", "interest": "관심도", "grade": "A|B|C"},
                 risk="notify", requires_contact=True, fn=record_lead),
        ToolSpec("transfer_to_agent", "사람 상담원 연결 (요약을 함께 넘김)", {"reason": "사유", "summary": "통화 요약"},
                 risk="handoff", fn=transfer_to_agent),
        ToolSpec("end_call", "통화 종료", {"disposition": "통화 결과"}, choices={"disposition": DISPOSITIONS},
                 risk="handoff", fn=end_call),
    ])
