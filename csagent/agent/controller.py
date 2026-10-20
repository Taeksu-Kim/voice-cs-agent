"""C' — 세션 컨트롤러: 업무 배정·전환·대기열을 코드가 관리하고, 각 업무는 워커가 처리한다.

한 턴의 흐름
1) 진행 중인 업무가 없으면 업무를 정한다 — 아웃바운드는 처음부터 정해져 있고, 인바운드는 라우터가 판별
2) 해당 워커 정책으로 공통 대화 루프를 돈다 (허용 도구는 게이트가 강제)
3) 워커가 switch를 내면 현재 업무를 대기열에 넣고 라우터로 새 업무를 정해 한 번 더 돈다
4) 워커가 done을 내면 업무를 끝내고, 대기열에 남은 업무가 있으면 다음 턴에 이어간다
5) 대기열에 용건이 있는 동안은 매 턴 라우터로 고객이 그 용건을 꺼냈는지 본다. 꺼냈으면 코드가 전환한다
   (워커가 done을 제때 내 주기를 기대하지 않는다 — 업무를 마친 워커가 다음 용건까지 직접 답하던 문제)
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Optional

from csagent.agent.baseline import finish_turn, start_turn
from csagent.agent.contracts import AgentEvent, AgentReply, TurnInput
from csagent.agent.loop import LoopOutcome, TurnLoop
from csagent.agent.router import Router
from csagent.agent.state import CallState, TaskState
from csagent.agent.workers.specs import Worker
from csagent.obs import event

log = logging.getLogger("csagent.agent")

_ASK_AGAIN = "어떤 도움이 필요하신지 한 번만 더 말씀해 주시겠어요?"
# 턴 처리 중 예외(LLM·검색 서버 장애 등) — 고객에게 아무 말도 안 나가고 이력에서 고객 말이 빠지던 것을 막는다 (docs/09 G5)
_TURN_ERROR = "죄송해요, 잠시 문제가 생겼어요. 한 번만 다시 말씀해 주시겠어요?"
_TURN_ERROR_AGAIN = "계속 연결이 원활하지 않아서 상담원에게 연결해 드릴게요."


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
        try:
            yield from self._turn(state, turn, t0)
        except Exception as e:   # 이미 나간 말은 되돌릴 수 없다 — 사과 한마디로 턴을 닫고, 연속이면 상담원에게 넘긴다
            again = bool([x for x in state.trace.of("turn_error") if x.turn == state.turn - 1])
            state.trace.add(state.turn, "turn_error", {"error": f"{type(e).__name__}: {e}"[:300], "handoff": again})
            event(log, "turn_error", logging.ERROR, call_id=state.call_id, turn=state.turn,
                  error=f"{type(e).__name__}: {e}"[:300], handoff=again)
            text = _TURN_ERROR_AGAIN if again else _TURN_ERROR
            yield AgentEvent("delta", {"text": text})
            reply = AgentReply(text=text)
            if again:
                reply.end_call, reply.disposition = True, "transferred"
            yield finish_turn(state, turn, reply)

    def _turn(self, state: CallState, turn: TurnInput, t0: float) -> Iterator[AgentEvent]:
        if state.active is None:
            first, *rest = self._first_tasks(state, turn.utterance)
            self._assign(state, first)
            self._enqueue(state, rest)
        elif state.queue:
            self._take_up_queued(state, turn.utterance)

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
                new, *rest = self.router.route(state, turn.utterance, exclude=task.name)
                event(log, "task_switch", call_id=state.call_id, turn=state.turn, by="worker",
                      src=task.name, dst=new)
                state.queue.insert(0, task)            # 앞 업무는 끝나지 않았으니 대기열 맨 앞으로
                self._assign(state, new)
                self._enqueue(state, rest)
                continue
            if outcome.added:                          # 이번 업무는 계속하고, 덧붙인 용건은 대기열로
                self._enqueue(state, self.router.route(state, turn.utterance, exclude=task.name))
            break

        reply = outcome.reply
        if outcome.intent == "switch":                  # 두 번째 워커도 자기 업무가 아니라고 함
            event(log, "ask_again", logging.WARNING, call_id=state.call_id, turn=state.turn,
                  worker=state.active.name)
            reply.text = _ASK_AGAIN
            yield AgentEvent("delta", {"text": _ASK_AGAIN})
        elif outcome.intent == "done":
            state.active.done = True
            state.finished.append(state.active)
            state.trace.add(state.turn, "task_done", {"task": state.active.name})
            event(log, "task_done", call_id=state.call_id, turn=state.turn, task=state.active.name)
            state.active = state.queue.pop(0) if state.queue else None
            if state.active is not None:
                state.active.started = True
                state.trace.add(state.turn, "task_start", {"task": state.active.name})
        yield finish_turn(state, turn, reply)

    def _first_tasks(self, state: CallState, utterance: str) -> list[str]:
        preset = state.ctx.customer.get("outbound_task")
        if state.ctx.direction == "outbound" and preset in self.workers:
            state.trace.add(state.turn, "route", {"task": preset, "reason": "아웃바운드 지정 업무"})
            return [preset]
        return self.router.route(state, utterance)

    @staticmethod
    def _enqueue(state: CallState, names: list[str]) -> None:
        waiting = {t.name for t in state.queue} | ({state.active.name} if state.active else set())
        for name in names:
            if name not in waiting and name != "general":   # general은 진행 중 업무가 받으면 된다
                state.queue.append(TaskState(name))
                state.trace.add(state.turn, "task_queued", {"task": name})
                waiting.add(name)

    def _take_up_queued(self, state: CallState, utterance: str) -> None:
        """고객이 대기 중인 용건을 꺼냈으면 그 업무로 바꾼다. 하던 업무는 대기열 끝에 둔다 (끝났는지는 모르므로)."""
        names = self.router.route(state, utterance)
        target = next((t for t in state.queue if t.name == names[0]), None)
        if target is None or target.name == state.active.name:
            return
        state.queue.remove(target)
        state.active.started = True
        state.queue.append(state.active)
        state.trace.add(state.turn, "task_switch", {"from": state.active.name, "to": target.name})
        event(log, "task_switch", call_id=state.call_id, turn=state.turn, by="queue",
              src=state.active.name, dst=target.name)
        state.active = target
        target.started = True
        state.trace.add(state.turn, "task_start", {"task": target.name})

    @staticmethod
    def _assign(state: CallState, name: str) -> None:
        state.active = TaskState(name, started=True)
        # 같은 통화에서 끝냈던 업무가 다시 시작되면 그때 한 조회·슬롯을 이어받는다 (docs/09 G8).
        # 새로 시작하면 「첫 조회 전」으로 돌아가 절차 강제가 반복됐다 — 모델은 이전 결과를 알아서 다시 조회하지 않음
        # (s3_b6 r1: 같은 업무 재시작 → 답 4번 버림 → 「확인해 볼게요」만 들림, runs/20261019_202026_C)
        prev = next((t for t in reversed(state.finished) if t.name == name), None)
        if prev is not None:
            state.active.tools_done |= prev.tools_done
            state.active.tools_empty |= prev.tools_empty
            state.active.slots = {**prev.slots}
        state.trace.add(state.turn, "task_start", {"task": name, "resumed": prev is not None})
