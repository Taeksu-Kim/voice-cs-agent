"""기준선 에이전트 (A0 / A1) — 단일 LLM이 모든 도구를 들고 모든 문의를 처리한다.

- A0: gate=None — 실행 조건을 프롬프트로만 지킨다
- A1: gate=ExecutionGate() — 같은 프롬프트 + 실행 직전 코드 검사 (C'와 같은 게이트)
대화 루프는 C'의 워커와 같은 TurnLoop을 쓰고, 정책만 「모든 업무·모든 도구」로 고정한다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Iterator, Optional

from csagent.agent import prompts
from csagent.agent.contracts import AgentEvent, AgentReply, TurnInput
from csagent.agent.guardrails.gate import ExecutionGate
from csagent.agent.loop import LoopOutcome, TurnLoop, TurnPolicy
from csagent.agent.state import CallState
from csagent.agent.tools.registry import ToolRegistry
from csagent.llm import ChatLLM
from csagent.obs import event
from csagent.retrieval.retriever import HybridRetriever

log = logging.getLogger("csagent.agent")

BASELINE_POLICY = TurnPolicy("baseline", prompts.BASELINE_TASK)


class BaselineAgent:
    def __init__(self, retriever: HybridRetriever, llm: ChatLLM, tools: ToolRegistry, *,
                 gate: Optional[ExecutionGate] = None, **loop_kwargs) -> None:
        self.loop = TurnLoop(retriever, llm, tools, gate=gate, **loop_kwargs)
        self.name = "A1" if gate else "A0"

    def respond(self, state: CallState, turn: TurnInput) -> AgentReply:
        for event in self.respond_stream(state, turn):
            if event.type == "done":
                return event.data["reply"]
        raise RuntimeError("respond_stream이 done 없이 끝났다")

    def respond_stream(self, state: CallState, turn: TurnInput) -> Iterator[AgentEvent]:
        t0 = time.perf_counter()
        start_turn(state, turn)
        for item in self.loop.run(state, turn.utterance, BASELINE_POLICY, t0=t0):
            if isinstance(item, LoopOutcome):
                yield finish_turn(state, turn, item.reply)
            else:
                yield item


def start_turn(state: CallState, turn: TurnInput) -> None:
    state.turn += 1
    state.trace.step = 0
    state.trace.add(state.turn, "turn_start", {"utterance": turn.utterance, "event": turn.event})
    state.trace.add(state.turn, "state", state.snapshot())
    if state.turn == 1:
        event(log, "call_start", call_id=state.call_id, direction=state.ctx.direction,
              audience=state.ctx.audience)


def finish_turn(state: CallState, turn: TurnInput, reply: AgentReply) -> AgentEvent:
    state.history += [("customer", turn.utterance), ("agent", reply.text)]
    state.trace.add(state.turn, "reply", {"text": reply.text, "disposition": reply.disposition})
    _log_turn(state, reply)
    return AgentEvent("done", {"reply": reply})


def _log_turn(state: CallState, reply: AgentReply) -> None:
    """턴 지표 한 건 — 모니터링의 지연·토큰·응답 길이 집계는 이 이벤트로 한다."""
    llm = [e for e in state.trace.of("llm") if e.turn == state.turn]
    t = reply.timings_ms
    event(log, "turn", call_id=state.call_id, turn=state.turn,
          workers=list(dict.fromkeys(e.data.get("role") for e in llm)), llm_steps=len(llm),
          ms_total=round(t.get("total", 0)), ms_first_token=round(t.get("first_token", 0)),
          ms_llm=round(t.get("llm", 0)), ms_retrieve=round(t.get("retrieve", 0)),
          prompt_tokens=reply.tokens.get("prompt", 0), completion_tokens=reply.tokens.get("completion", 0),
          tools=[c.name for c in reply.tool_calls], chars=len(reply.text),
          disposition=reply.disposition, end_call=reply.end_call)


def reply_to_dict(reply: AgentReply) -> dict:
    """AgentReply → JSON 직렬화 가능한 dict (서버·결과 저장용)."""
    return asdict(reply)
