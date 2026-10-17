"""라우터 — 새 업무를 판별한다. 진행 중인 업무가 없을 때, 워커가 switch·add를 냈을 때만 불린다.
한 말에 용건이 여럿이면 말한 순서대로 모두 돌려준다 (컨트롤러가 첫째를 맡기고 나머지는 대기열에 넣는다).

업무 판별은 말의 의미 판단이라 LLM이 한다 (짧은 비스트리밍 호출).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from csagent.agent.state import CallState
from csagent.agent.workers.specs import Worker
from csagent.llm import ChatLLM, parse_json
from csagent.obs import event

log = logging.getLogger("csagent.agent")

_PROMPT = """\
고객센터 통화에서 고객의 용건이 어느 업무인지 고른다.

[업무 목록]
{tasks}

[최근 대화]
{history}

[고객의 이번 말]
{utterance}

[출력] JSON 하나: {{"tasks": ["업무 이름", ...], "reason": "한 문장"}}
- tasks: 이번 말에 들어 있는 용건을 말한 순서대로. 대부분 하나다. 용건이 분명히 둘 이상일 때만 여럿을 넣는다.
- 애매하면 general을 고른다."""


class Router:
    def __init__(self, llm: ChatLLM, workers: dict[str, Worker]) -> None:
        self.llm = llm
        self.workers = workers

    def route(self, state: CallState, utterance: str, exclude: Optional[str] = None) -> list[str]:
        """이번 말의 업무들 (말한 순서, 최소 하나)."""
        candidates = {n: w for n, w in self.workers.items() if w.inbound and n != exclude}
        history = "\n".join(f"{'고객' if r == 'customer' else '상담원'}: {t}" for r, t in state.history[-4:])
        prompt = _PROMPT.format(
            tasks="\n".join(f"- {n}: {w.description}" for n, w in candidates.items()),
            history=history or "(없음)", utterance=utterance)
        res = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=100, json_mode=True)
        try:
            out = parse_json(res.text)
            raw = out.get("tasks") or [out.get("task")]
            tasks = list(dict.fromkeys(t for t in raw if t in candidates))[:3]   # 목록에 있는 이름만, 중복 제거
            reason = out.get("reason", "")
        except (ValueError, json.JSONDecodeError, TypeError):
            tasks, reason = [], "판별 실패"
            event(log, "route_failed", logging.WARNING, call_id=state.call_id, turn=state.turn,
                  llm_call_id=res.call_id, output=res.text[:300])
        tasks = tasks or ["general"]
        state.trace.add(state.turn, "route", {"task": tasks[0], "tasks": tasks, "reason": reason,
                                              "exclude": exclude, "ms": round(res.latency_ms),
                                              "call_id": res.call_id})
        event(log, "route", call_id=state.call_id, turn=state.turn, tasks=tasks, exclude=exclude,
              ms=round(res.latency_ms), llm_call_id=res.call_id)
        return tasks
