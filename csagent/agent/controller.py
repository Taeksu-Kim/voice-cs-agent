"""C' — 세션 컨트롤러: 업무 배정·전환·대기열을 코드가 관리하고, 각 업무는 워커가 처리한다.

한 턴의 흐름
1) 진행 중인 업무가 없으면 업무를 정한다 — 아웃바운드는 처음부터 정해져 있고, 인바운드는 라우터가 판별
2) 해당 워커 정책으로 공통 대화 루프를 돈다 (허용 도구는 게이트가 강제)
3) 워커가 switch를 내면 현재 업무를 대기열에 넣고 라우터로 새 업무를 정해 한 번 더 돈다
4) 워커가 done을 내면 업무를 끝내고, 대기열에 남은 업무가 있으면 다음 턴에 이어간다
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

from csagent.agent.baseline import finish_turn, start_turn
from csagent.agent.contracts import AgentEvent, AgentReply, TurnInput
from csagent.agent.loop import LoopOutcome, TurnLoop
from csagent.agent.router import Router
from csagent.agent.state import CallState, TaskState
from csagent.agent.workers.specs import Worker

_ASK_AGAIN = "어떤 도움이 필요하신지 한 번만 더 말씀해 주시겠어요?"


class CallController:
    name = "C'"

    def __init__(self, loop: TurnLoop, router: Router, workers: dict[str, Worker]) -> None:
        self.loop = loop
        self.router = router
        self.workers = workers

    def respond(self, state: CallState, turn: TurnInput) -> AgentReply:
        for event in self.respond_stream(state, turn):
            if event.type == "done":
                return event.data["reply"]
        raise RuntimeError("respond_stream이 done 없이 끝났다")

    def respond_stream(self, state: CallState, turn: TurnInput) -> Iterator[AgentEvent]:
        t0 = time.perf_counter()
        start_turn(state, turn)
        if state.active is None:
            self._assign(state, self._first_task(state, turn.utterance))

        outcome: Optional[LoopOutcome] = None
        for attempt in range(2):  # 전환은 한 턴에 한 번까지
            task = state.active
            yield AgentEvent("task", {"task": task.name, "attempt": attempt})
            for item in self.loop.run(state, turn.utterance, self.workers[task.name].policy,
                                      slots=task.slots, t0=t0):
                if isinstance(item, LoopOutcome):
                    outcome = item
                else:
                    yield item
            task.slots.update(outcome.slots)
            if outcome.intent == "switch" and attempt == 0:
                new = self.router.route(state, turn.utterance, exclude=task.name)
                state.queue.append(task)               # 앞 업무는 끝나지 않았으니 대기열로
                self._assign(state, new)
                continue
            break

        reply = outcome.reply
        if outcome.intent == "switch":                  # 두 번째 워커도 자기 업무가 아니라고 함
            reply.text = _ASK_AGAIN
            yield AgentEvent("delta", {"text": _ASK_AGAIN})
        elif outcome.intent == "done":
            state.active.done = True
            state.trace.add(state.turn, "task_done", {"task": state.active.name})
            state.active = state.queue.pop(0) if state.queue else None
        yield finish_turn(state, turn, reply)

    def _first_task(self, state: CallState, utterance: str) -> str:
        preset = state.ctx.customer.get("outbound_task")
        if state.ctx.direction == "outbound" and preset in self.workers:
            state.trace.add(state.turn, "route", {"task": preset, "reason": "아웃바운드 지정 업무"})
            return preset
        return self.router.route(state, utterance)

    @staticmethod
    def _assign(state: CallState, name: str) -> None:
        state.active = TaskState(name)
        state.trace.add(state.turn, "task_start", {"task": name})
