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
    "confirmation_required": "같은 대상으로 고객 재확인(request_confirmation)을 받은 뒤에만 실행할 수 있습니다",
    "duplicate": "이미 같은 대상으로 실행했습니다",
    "contact_required": "통화 상대가 담당자로 확인된 뒤에만 쓸 수 있습니다 (ask_contact)",
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
            return GateDecision(False, "confirmation_required")
        if spec.risk == "execute" and execution_key(spec, args) in state.executed:
            return GateDecision(False, "duplicate")
        return GateDecision(True)
