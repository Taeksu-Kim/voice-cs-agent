"""에이전트 입·출력 계약.

음성을 나중에 붙이기 위한 경계면 (docs/01 §5):
- 입력은 처음부터 STT 출력 형태 (발화 텍스트 + 이벤트 + 지연 예산)
- 출력은 TTS로 읽힐 짧은 문장 + 도구 호출 + 단계별 시간
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Event = Literal["speech", "barge_in", "silence", "hangup"]
Disposition = Literal["resolved", "link_sent", "transferred", "callback", "refused", "unresolved"]


@dataclass
class CallContext:
    """통화 시작 시 주어지는 것 (요청 경로로 확정되는 값)."""
    direction: Literal["inbound", "outbound"] = "inbound"
    audience: str = "consumer"
    companies: tuple[str, ...] = ()
    customer: dict[str, Any] = field(default_factory=dict)  # 가짜 CRM: 이름·주문 등
    today: str = ""                                          # 시나리오 기준 날짜


@dataclass
class TurnInput:
    utterance: str
    event: Event = "speech"
    t_budget_ms: Optional[int] = None


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result: Any = None


@dataclass
class AgentEvent:
    """스트리밍 이벤트: retrieval | delta | tool_call | tool_result | done."""
    type: str
    data: dict[str, Any]


@dataclass
class AgentReply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    end_call: bool = False
    disposition: Optional[Disposition] = None
    retrieved: list[str] = field(default_factory=list)      # 검색된 chunk_id
    timings_ms: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
