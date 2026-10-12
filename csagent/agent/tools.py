"""상담 도구 (가짜 구현). 실제 시스템 대신 CallContext.customer의 가짜 CRM을 읽고 쓴다.

도구 설명(TOOL_SPECS)은 프롬프트에 그대로 들어간다.
"""

from __future__ import annotations

from typing import Any, Callable

from csagent.agent.contracts import CallContext

TOOL_SPECS = """\
- lookup_orders(): 지금 통화 중인 본인의 토스쇼핑 주문 목록과 배송 상태를 조회한다. 다른 사람 주문은 조회할 수 없다.
- send_app_link(target): 고객 휴대폰으로 토스 앱 화면 링크를 문자로 보낸다. target 예: "주문 내역", "착오송금 접수".
- transfer_to_agent(reason): 사람 상담원에게 연결한다. 에이전트가 해결할 수 없거나 고객이 원할 때.
- end_call(disposition): 통화를 끝낸다. disposition = resolved | link_sent | transferred | refused | unresolved"""


class ToolBox:
    def __init__(self, ctx: CallContext) -> None:
        self.ctx = ctx
        self._tools: dict[str, Callable[..., Any]] = {
            "lookup_orders": self.lookup_orders,
            "send_app_link": self.send_app_link,
            "transfer_to_agent": self.transfer_to_agent,
            "end_call": self.end_call,
        }

    def call(self, name: str, args: dict[str, Any]) -> Any:
        if name not in self._tools:
            return {"error": f"알 수 없는 도구: {name}"}
        try:
            return self._tools[name](**args)
        except TypeError as e:
            return {"error": f"잘못된 인자: {e}"}

    def lookup_orders(self) -> dict:
        return {"today": self.ctx.today, "orders": self.ctx.customer.get("orders", [])}

    def send_app_link(self, target: str) -> dict:
        return {"sent": True, "target": target}

    def transfer_to_agent(self, reason: str = "") -> dict:
        return {"transferred": True, "reason": reason}

    def end_call(self, disposition: str = "resolved") -> dict:
        return {"ended": True, "disposition": disposition}
