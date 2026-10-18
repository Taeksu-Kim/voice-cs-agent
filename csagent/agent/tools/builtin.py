"""에이전트 내장 도구 — 업무 데이터가 아니라 통화 자체를 제어한다 (docs/08 6-1). MCP를 거치지 않는다.

실서비스에서는 콜 인프라(상담원 연결·통화 종료)와 대화 상태(담당자 확인 질문)에 붙는 기능이다.
게이트 메타데이터(risk·asks_customer 등)도 여기 코드에 둔다.
"""

from __future__ import annotations

from csagent.agent.state import CallState
from csagent.agent.tools.registry import ToolSpec


def ask_contact(state: CallState) -> dict:
    """아웃바운드: 통화 상대가 담당자인지 묻는다. 확정은 다음 턴 고객 발화 판정으로."""
    if state.contact_verified:
        return {"status": "already_verified", "note": "이미 담당자로 확인됐습니다. 다시 묻지 마세요."}
    state.ask("contact")
    return {"status": "pending", "note": "담당자 본인인지 물어보세요."}


def transfer_to_agent(state: CallState, reason: str = "", summary: str = "") -> dict:
    return {"transferred": True, "reason": reason, "summary": summary}


DISPOSITIONS = ("resolved", "link_sent", "transferred", "refused", "unresolved")


def end_call(state: CallState, disposition: str = "resolved") -> dict:
    if disposition not in DISPOSITIONS:
        return {"error": f"bad_disposition:{disposition} (허용: {', '.join(DISPOSITIONS)})"}
    return {"ended": True, "disposition": disposition}


def builtin_specs() -> list[ToolSpec]:
    return [
        ToolSpec("ask_contact", "아웃바운드: 통화 상대가 담당자인지 확인 질문을 건다", risk="notify",
                 asks_customer=True, fn=ask_contact),
        ToolSpec("transfer_to_agent", "사람 상담원 연결 (요약을 함께 넘김)", {"reason": "사유", "summary": "통화 요약"},
                 risk="handoff", fn=transfer_to_agent),
        ToolSpec("end_call", "통화 종료", {"disposition": "통화 결과"}, choices={"disposition": DISPOSITIONS},
                 risk="handoff", fn=end_call),
    ]
