"""에이전트 구조(A0 / A1 / C')가 공통으로 구현하는 인터페이스.

평가 파이프라인과 웹 서버는 이 인터페이스만 보므로, 구조를 바꿔도 그대로 돌아간다.
"""

from __future__ import annotations

import copy
from typing import Iterator, Protocol

from csagent.agent.contracts import AgentEvent, AgentReply, CallContext, TurnInput
from csagent.agent.prompts import GREETING
from csagent.agent.state import CallState


class Agent(Protocol):
    name: str

    def respond_stream(self, state: CallState, turn: TurnInput) -> Iterator[AgentEvent]: ...

    def respond(self, state: CallState, turn: TurnInput) -> AgentReply: ...


def new_call(ctx: CallContext) -> CallState:
    """시나리오의 통화 맥락으로 새 통화 상태를 만든다. 가짜 도구가 고객 데이터를 바꾸므로 깊은 복사.

    인바운드는 고객이 말하기 전에 고정 인사(AI 상담원·녹음 안내)를 0턴 발화로 넣는다.
    아웃바운드는 받는 쪽이 먼저 「여보세요」 하고 에이전트가 첫 응답에서 소개한다 (워커 지시문).
    """
    state = CallState(ctx=copy.deepcopy(ctx))
    if ctx.direction == "inbound":
        text = GREETING.format(service=ctx.service)
        state.history.append(("agent", text))
        state.trace.add(0, "greeting", {"text": text})
    return state
