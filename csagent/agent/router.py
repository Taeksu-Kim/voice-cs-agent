"""라우터 — 새 업무를 판별한다. 진행 중인 업무가 없을 때와 워커가 switch를 냈을 때만 불린다.

업무 판별은 말의 의미 판단이라 LLM이 한다 (짧은 비스트리밍 호출).
"""

from __future__ import annotations

import json
from typing import Optional

from csagent.agent.state import CallState
from csagent.agent.workers.specs import Worker
from csagent.llm import ChatLLM, parse_json

_PROMPT = """\
고객센터 통화에서 고객의 용건이 어느 업무인지 고른다.

[업무 목록]
{tasks}

[최근 대화]
{history}

[고객의 이번 말]
{utterance}

[출력] JSON 하나: {{"task": "업무 이름", "reason": "한 문장"}}
애매하면 general을 고른다."""


class Router:
    def __init__(self, llm: ChatLLM, workers: dict[str, Worker]) -> None:
        self.llm = llm
        self.workers = workers

    def route(self, state: CallState, utterance: str, exclude: Optional[str] = None) -> str:
        candidates = {n: w for n, w in self.workers.items() if w.inbound and n != exclude}
        history = "\n".join(f"{'고객' if r == 'customer' else '상담원'}: {t}" for r, t in state.history[-4:])
        prompt = _PROMPT.format(
            tasks="\n".join(f"- {n}: {w.description}" for n, w in candidates.items()),
            history=history or "(없음)", utterance=utterance)
        res = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=80, json_mode=True)
        try:
            out = parse_json(res.text)
            task = out.get("task") if out.get("task") in candidates else "general"
            reason = out.get("reason", "")
        except (ValueError, json.JSONDecodeError):
            task, reason = "general", "판별 실패"
        state.trace.add(state.turn, "route", {"task": task, "reason": reason, "exclude": exclude,
                                              "ms": round(res.latency_ms)})
        return task
