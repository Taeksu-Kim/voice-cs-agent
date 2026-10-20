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
- state        턴 시작 때 통화 상태 스냅샷 (재현기가 이 턴부터 다시 시작할 때 쓴다)
- call_open    통화 시작 때 도메인 데이터 (고객·주문·카드 픽스처)

좌표 (재현·그림용): 이벤트마다 at = "t<턴>.s<스텝>.<종류>" (같은 스텝에 같은 종류가 또 나오면 "#2"),
where = "<업무>/<프로토콜 단계>". 스텝 0은 LLM 전 (라우팅·검색), 1부터 LLM 스텝 (한 턴 안에서 워커가 바뀌어도 이어 센다).
통화 전체 주소는 "<call_id>@<at>" — scripts/trace_graph.py가 그린 mermaid 노드 id가 이 좌표다.
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
    step: int = 0
    at: str = ""
    where: str = ""


@dataclass
class Trace:
    events: list[TraceEvent] = field(default_factory=list)
    step: int = 0       # 지금 스텝 (start_turn이 0으로, 루프가 LLM 스텝마다 올린다)
    where: str = ""     # 지금 업무/단계 (루프가 갱신)
    _seen: dict[str, int] = field(default_factory=dict, repr=False)

    def add(self, turn: int, kind: str, data: dict[str, Any]) -> None:
        at = f"t{turn}.s{self.step}.{kind}"
        n = self._seen[at] = self._seen.get(at, 0) + 1
        self.events.append(TraceEvent(turn, kind, data, step=self.step, at=at if n == 1 else f"{at}#{n}",
                                      where=self.where))

    def of(self, kind: str) -> list[TraceEvent]:
        return [e for e in self.events if e.kind == kind]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self.events]
