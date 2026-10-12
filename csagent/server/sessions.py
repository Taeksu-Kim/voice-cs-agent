"""통화 세션 저장소 (메모리). 세션 = 통화 1건: 통화 상태(CallState) + 턴별 응답 메타."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from csagent.agent.contracts import AgentReply, CallContext
from csagent.agent.interface import new_call
from csagent.agent.state import CallState


@dataclass
class Session:
    id: str
    state: CallState
    scenario_id: str = ""
    replies: list[AgentReply] = field(default_factory=list)
    disposition: Optional[str] = None
    ended: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def record(self, reply: AgentReply) -> None:
        """대화 이력은 에이전트가 state에 기록한다 — 세션은 응답 메타와 종료 여부만."""
        self.replies.append(reply)
        self.disposition = reply.disposition or self.disposition
        self.ended = self.ended or reply.end_call


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, ctx: CallContext, scenario_id: str = "") -> Session:
        s = Session(id=uuid.uuid4().hex[:12], state=new_call(ctx), scenario_id=scenario_id)
        with self._lock:
            self._sessions[s.id] = s
        return s

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)
