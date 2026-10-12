"""통화 추적 로그 — 평가 파이프라인의 결정적 검사(절차 위반 등)가 이것을 읽는다.

이벤트 종류
- turn_start   고객 발화
- retrieval    검색 결과
- llm          LLM 호출 (역할·토큰·시간)
- pending        확인 질문을 걸었음
- pending_answer LLM이 판정한 고객 답 + 인용 대조 결과(verified)
- tool_request 에이전트가 요청한 도구
- gate         실행 게이트 판정 (허용 여부·사유)
- tool_result  실행 결과 (성공·실패)
- guardrail    가드레일 발동 (층·규칙·조치)
- reply        최종 응답
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    turn: int
    kind: str
    data: dict[str, Any]
    t: float = field(default_factory=time.time)


@dataclass
class Trace:
    events: list[TraceEvent] = field(default_factory=list)

    def add(self, turn: int, kind: str, data: dict[str, Any]) -> None:
        self.events.append(TraceEvent(turn, kind, data))

    def of(self, kind: str) -> list[TraceEvent]:
        return [e for e in self.events if e.kind == kind]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self.events]
