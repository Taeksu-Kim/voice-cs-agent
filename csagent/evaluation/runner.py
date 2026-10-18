"""시나리오 1건 실행: 고객 시뮬레이터 ↔ 에이전트를 번갈아 돌려 통화 기록(Transcript)을 만든다."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from csagent.agent.contracts import AgentReply, TurnInput
from csagent.agent.interface import Agent, new_call
from csagent.agent.state import CallState
from csagent.evaluation.scenario import Scenario
from csagent.evaluation.simulator import CustomerSimulator
from csagent.obs import event

log = logging.getLogger("csagent.eval")


@dataclass
class Transcript:
    scenario_id: str
    state: CallState
    agent_name: str = ""
    replies: list[AgentReply] = field(default_factory=list)      # 에이전트 턴별 메타 (도구·검색·시간)
    disposition: Optional[str] = None     # 에이전트가 남긴 통화 결과. 남기지 않고 고객이 끊으면 "customer_ended"
    end_reason: str = ""                                          # agent_ended | customer_hung_up | max_turns
    customer_satisfied: Optional[bool] = None                     # 시뮬레이터 자기 보고 (참고용, 채점에 안 씀)

    @property
    def turns(self) -> list[tuple[str, str]]:
        return self.state.history

    def tool_names(self) -> list[str]:
        return [c.name for r in self.replies for c in r.tool_calls]

    def retrieved_docs(self) -> set[str]:
        return {cid.split("#")[0] for r in self.replies for cid in r.retrieved}

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id, "agent": self.agent_name,
            "disposition": self.disposition, "end_reason": self.end_reason,
            "customer_satisfied": self.customer_satisfied, "turns": self.turns,
            "replies": [{"text": r.text, "tools": [(c.name, c.args, c.result) for c in r.tool_calls],
                         "retrieved": r.retrieved, "timings_ms": r.timings_ms, "tokens": r.tokens}
                        for r in self.replies],
            "trace": self.state.trace.to_dicts(),
        }


def run_scenario(agent: Agent, customer: CustomerSimulator, scenario: Scenario, stack: Any = None) -> Transcript:
    """stack(csagent/stack.py)을 주면 콜 인프라로 통화를 연다 — 시나리오 업무 데이터를 도메인 API에 심고 끝나면 지운다."""
    state = stack.open(scenario.context) if stack is not None else new_call(scenario.context)
    tr = Transcript(scenario.id, state, agent_name=getattr(agent, "name", ""))
    try:
        return _converse(agent, customer, scenario, tr)
    finally:
        if stack is not None:
            stack.close(state)


def _converse(agent: Agent, customer: CustomerSimulator, scenario: Scenario, tr: Transcript) -> Transcript:
    for _ in range(scenario.max_turns):
        utterance = customer.next(scenario, tr.state.history)
        if utterance is None:
            tr.end_reason = "customer_hung_up"
            break
        reply = agent.respond(tr.state, TurnInput(utterance))
        tr.replies.append(reply)
        if reply.disposition:
            tr.disposition = reply.disposition  # 링크 발송 등 중간 결과도 남기고, 마지막 값이 최종
        if reply.end_call:
            tr.end_reason = "agent_ended"
            break
    else:
        tr.end_reason = "max_turns"
    tr.customer_satisfied = getattr(customer, "satisfied", None)
    if tr.disposition is None:
        # 에이전트가 결과를 남기지 않고 끝난 통화 — 해결 여부는 시뮬레이터 자기 보고가 아니라 Judge가 판정한다
        tr.disposition = "customer_ended"
    event(log, "call_end", call_id=tr.state.call_id, disposition=tr.disposition, end_reason=tr.end_reason,
          turns=len(tr.replies))
    return tr
