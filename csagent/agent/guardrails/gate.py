"""공통 실행 게이트 — 도구 실행 직전에 도구 선언(requires_*)과 통화 상태를 대조한다.

A1·C'가 같은 게이트를 쓴다 (구조의 효과만 비교하기 위해). A0는 게이트를 끈 구성.
막히면 사유 코드를 돌려주고, 에이전트는 그 사유를 보고 다음 행동(인증 요청·재확인 등)을 정한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from csagent.agent.state import CallState
from csagent.agent.tools.registry import ToolSpec, execution_key

# 사유 코드 → 에이전트에게 돌려줄 설명
REASONS = {
    "not_allowed": "이 업무에서 쓸 수 없는 도구입니다",
    "auth_required": "본인 인증(request_app_auth)이 먼저 필요합니다",
    "confirmation_required": "같은 대상으로 고객 동의를 받은 뒤에만 실행할 수 있습니다",
    "duplicate": "이미 같은 대상으로 실행했습니다",
    "tasks_queued": ("고객이 꺼낸 다른 용건이 아직 처리되지 않았습니다. 이번에는 tool 없이, 그 용건도 이어서 도와드릴지 "
                     "고객에게 물어보세요. 고객이 그래도 끝내겠다고 하면 다음 턴에 end_call로 마칩니다"),
    "confirmed_not_executed": ("고객이 동의한 실행이 아직 성공하지 않았습니다. 실행하지 않았으면 먼저 실행하고, "
                               "실패했으면 처리됐다고 하지 말고 transfer_to_agent로 연결하세요"),
    "contact_required": "통화 상대가 담당자로 확인된 뒤에만 쓸 수 있습니다 (ask_contact)",
    "result_unheard": "도구로 처리한 결과(실행·발송)를 알린 턴에는 통화를 끝내지 않습니다 — 고객 대답을 들은 뒤에 끝냅니다",
}


@dataclass
class GateDecision:
    allowed: bool
    reason: str = ""

    @property
    def message(self) -> str:
        return REASONS.get(self.reason, "")


class ExecutionGate:
    def check(self, state: CallState, spec: ToolSpec, args: dict[str, Any],
              allowed_tools: Optional[set[str]] = None) -> GateDecision:
        if allowed_tools is not None and spec.name not in allowed_tools:
            return GateDecision(False, "not_allowed")
        if spec.requires_contact and not state.contact_verified:
            return GateDecision(False, "contact_required")
        if spec.requires_auth and not state.authenticated:
            return GateDecision(False, "auth_required")
        if spec.requires_confirmation and not state.is_confirmed(spec.name, spec.target(args)):
            return GateDecision(False, "confirmation_required")   # 루프가 확인 질문을 건다
        if spec.risk == "execute" and execution_key(spec, args) in state.executed:
            return GateDecision(False, "duplicate")
        if spec.name == "end_call" and args.get("disposition") not in ("refused", "transferred") \
                and any(not t.started for t in state.queue) \
                and (state.end_blocked_turn is None or state.end_blocked_turn == state.turn):
            # 고객이 꺼냈는데 아무도 맡지 않은 용건이 있으면 한 번 막는다. 남은 용건을 들은 뒤에도 끝내겠다면 허용
            return GateDecision(False, "tasks_queued")
        if spec.name == "end_call" and args.get("disposition", "resolved") in ("resolved", "link_sent") \
                and state.unexecuted_confirmations():
            # 동의받은 실행이 성공하지 않았는데 「해결」로 끊는 것을 막는다 (기록 대조 — 실패했으면 이관으로)
            return GateDecision(False, "confirmed_not_executed")
        if spec.name == "end_call" and args.get("disposition", "resolved") in ("resolved", "link_sent") \
                and state.acted_turn == state.turn:
            # 처리 결과를 알리면서 「다른 도움 필요하시면」과 종료를 같이 내 고객이 다음 용건을 못 꺼냄
            # (h4 카드 실행, 10/17 → 같은 날 s4_b0_friend 링크 발송에서도 확인해 notify까지 넓힘)
            return GateDecision(False, "result_unheard")
        return GateDecision(True)
