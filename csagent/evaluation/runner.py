"""시나리오 1건 실행: 고객 시뮬레이터 ↔ 에이전트를 번갈아 돌려 통화 기록(Transcript)을 만든다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from csagent.agent.agent import CSAgent
from csagent.agent.contracts import AgentReply, TurnInput
from csagent.evaluation.scenario import Scenario
from csagent.evaluation.simulator import CustomerSimulator


@dataclass
class Transcript:
    scenario_id: str
    turns: list[tuple[str, str]] = field(default_factory=list)   # [("customer"|"agent", 발화)]
    replies: list[AgentReply] = field(default_factory=list)      # 에이전트 턴별 메타 (도구·검색·시간)
    disposition: Optional[str] = None
    end_reason: str = ""                                          # agent_ended | customer_hung_up | max_turns

    def tool_names(self) -> list[str]:
        return [c.name for r in self.replies for c in r.tool_calls]

    def retrieved_docs(self) -> set[str]:
        return {cid.split("#")[0] for r in self.replies for cid in r.retrieved}

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id, "disposition": self.disposition,
            "end_reason": self.end_reason, "turns": self.turns,
            "replies": [{"text": r.text, "tools": [(c.name, c.args, c.result) for c in r.tool_calls],
                         "retrieved": r.retrieved, "timings_ms": r.timings_ms, "tokens": r.tokens}
                        for r in self.replies],
        }


def run_scenario(agent: CSAgent, customer: CustomerSimulator, scenario: Scenario) -> Transcript:
    tr = Transcript(scenario.id)
    for _ in range(scenario.max_turns):
        utterance = customer.next(scenario, tr.turns)
        if utterance is None:
            tr.end_reason = "customer_hung_up"
            break
        reply = agent.respond(scenario.context, tr.turns, TurnInput(utterance))
        tr.turns += [("customer", utterance), ("agent", reply.text)]
        tr.replies.append(reply)
        if reply.disposition:
            tr.disposition = reply.disposition  # 링크 발송 등 중간 결과도 남기고, 마지막 값이 최종
        if reply.end_call:
            tr.end_reason = "agent_ended"
            break
    else:
        tr.end_reason = "max_turns"
    if tr.disposition is None:
        # 에이전트가 결과를 남기지 않고 끝난 통화 — 고객이 만족하고 끊었을 때만 해결로 본다
        tr.disposition = "resolved" if getattr(customer, "satisfied", None) else "unresolved"
    return tr
