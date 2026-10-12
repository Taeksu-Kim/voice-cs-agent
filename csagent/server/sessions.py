"""통화 세션 저장소 (메모리). 세션 = 통화 1건: 통화 맥락 + 대화 이력 + 턴별 응답 메타."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from csagent.agent.contracts import AgentReply, CallContext


@dataclass
class Session:
    id: str
    ctx: CallContext
    scenario_id: str = ""
    history: list[tuple[str, str]] = field(default_factory=list)   # [("customer"|"agent", 발화)]
    replies: list[AgentReply] = field(default_factory=list)
    disposition: Optional[str] = None
    ended: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def record(self, utterance: str, reply: AgentReply) -> None:
        self.history += [("customer", utterance), ("agent", reply.text)]
        self.replies.append(reply)
        self.disposition = reply.disposition or self.disposition
        self.ended = self.ended or reply.end_call


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, ctx: CallContext, scenario_id: str = "") -> Session:
        s = Session(id=uuid.uuid4().hex[:12], ctx=ctx, scenario_id=scenario_id)
        with self._lock:
            self._sessions[s.id] = s
        return s

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)
