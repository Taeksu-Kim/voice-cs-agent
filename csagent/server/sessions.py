"""통화 세션 저장소 (메모리). 세션 = 통화 1건: 통화 상태(CallState) + 턴별 응답 메타."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from csagent.agent.contracts import AgentReply, CallContext
from csagent.agent.interface import new_call
from csagent.agent.state import CallState


@dataclass
class Session:
    id: str
    state: CallState
    scenario_id: str = ""
    mode: str = "text"           # text(텍스트 콘솔) | voice(실제 테스트 탭, 마이크) — 로그·화면에서 구분
    situation: str = ""          # 실제 테스트 탭에서 고른 상황 카드 id (녹화 목록에 표시)
    replies: list[AgentReply] = field(default_factory=list)
    disposition: Optional[str] = None
    ended: bool = False
    saved: bool = False
    voice_turns: list[dict] = field(default_factory=list)   # 음성 턴별 STT 결과·문장·지연 (데모 재생 원본)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def record(self, reply: AgentReply) -> None:
        """대화 이력은 에이전트가 state에 기록한다 — 세션은 응답 메타와 종료 여부만."""
        self.replies.append(reply)
        self.disposition = reply.disposition or self.disposition
        self.ended = self.ended or reply.end_call


class SessionStore:
    def __init__(self, opener: Callable[[CallContext], CallState] = new_call) -> None:
        """opener: 통화를 여는 함수 — 도구 스택을 쓰면 stack.open (업무 데이터를 도메인 API에 심는다)."""
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._opener = opener

    def create(self, ctx: CallContext, scenario_id: str = "", mode: str = "text", situation: str = "") -> Session:
        state = self._opener(ctx)
        s = Session(id=state.call_id, state=state, scenario_id=scenario_id, mode=mode, situation=situation)   # 세션 id = 통화 id (로그와 같은 값)
        with self._lock:
            self._sessions[s.id] = s
        return s

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)
