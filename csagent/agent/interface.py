"""에이전트 구조(A0 / A1 / C')가 공통으로 구현하는 인터페이스.

평가 파이프라인과 웹 서버는 이 인터페이스만 보므로, 구조를 바꿔도 그대로 돌아간다.
"""

from __future__ import annotations

import copy
from typing import Iterator, Protocol

from csagent.agent.contracts import AgentEvent, AgentReply, CallContext, TurnInput
from csagent.agent.state import CallState


class Agent(Protocol):
    name: str

    def respond_stream(self, state: CallState, turn: TurnInput) -> Iterator[AgentEvent]: ...

    def respond(self, state: CallState, turn: TurnInput) -> AgentReply: ...


def new_call(ctx: CallContext) -> CallState:
    """시나리오의 통화 맥락으로 새 통화 상태를 만든다. 가짜 도구가 고객 데이터를 바꾸므로 깊은 복사."""
    return CallState(ctx=copy.deepcopy(ctx))
