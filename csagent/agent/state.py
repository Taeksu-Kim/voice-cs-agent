"""통화 상태 — 한 통화 동안 바뀌는 모든 것. 에이전트 구조(A0/A1/C')와 무관한 공통 계층.

실행 안전에 관한 상태(인증, 재확인, 담당자 확인, 실행 이력)는 LLM이 직접 바꾸지 못한다.
- 인증: 앱 인증 도구의 결과로만 바뀐다
- 재확인·담당자 확인: 에이전트가 질문을 걸어 두면(pending), 다음 턴에 LLM이 고객의 답(yes/no/unclear)과
  그 근거 인용을 낸다. 코드는 「질문이 이전 턴에 있었나」「인용이 실제 고객 발화에 있나」만 대조해 확정한다.
  (말의 의미 판단은 LLM, 값 대조는 코드 — 정규식으로 의미를 판단하지 않는다)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from csagent.agent.contracts import CallContext
from csagent.agent.trace import Trace

PendingKind = Literal["confirm", "contact"]


@dataclass
class Pending:
    """고객의 예/아니오를 기다리는 질문."""
    kind: PendingKind
    turn: int
    action: str = ""                      # confirm: 실행하려는 도구
    target: dict[str, Any] = field(default_factory=dict)


@dataclass
class Confirmation:
    action: str
    target: dict[str, Any]
    turn: int


def covers(asked: dict[str, Any], target: dict[str, Any]) -> bool:
    """확인받은 대상이 실행 대상의 키·값을 모두 포함하나 (확인 때 카드 이름 같은 설명 키가 더 붙어도 된다)."""
    return bool(target) and all(v is not None and asked.get(k) == v for k, v in target.items())


@dataclass
class TaskState:
    """업무 하나의 진행 상태 (C'의 워커가 사용)."""
    name: str
    stage: str = "start"
    slots: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    tools_done: set[str] = field(default_factory=set)   # 이 업무에서 성공한 도구 (절차 강제용)


@dataclass
class CallState:
    ctx: CallContext
    history: list[tuple[str, str]] = field(default_factory=list)   # [("customer"|"agent", 발화)]
    turn: int = 0
    # 실행 안전 상태
    authenticated: bool = False
    auth_failures: int = 0
    contact_verified: bool = False         # 아웃바운드: 통화 상대가 담당자로 확인됨
    pending: Optional[Pending] = None
    confirmations: list[Confirmation] = field(default_factory=list)
    executed: set[str] = field(default_factory=set)                # 실행 키 (중복 실행 방지)
    succeeded: list[tuple[str, dict]] = field(default_factory=list)  # 성공한 실행 (도구, 대상)
    # 업무 진행 (C')
    active: Optional[TaskState] = None
    queue: list[TaskState] = field(default_factory=list)
    trace: Trace = field(default_factory=Trace)

    # -- 재확인 --

    def ask(self, kind: PendingKind, action: str = "", target: Optional[dict] = None) -> None:
        self.pending = Pending(kind, self.turn, action, dict(target or {}))
        self.trace.add(self.turn, "pending", {"kind": kind, "action": action, "target": target or {}})

    def resolve_pending(self, answer: str, quote: str, utterance: str) -> bool:
        """LLM이 판정한 고객 답으로 걸어 둔 질문을 확정한다. 확정했으면 True.

        answer: "yes" | "no" | "unclear" (LLM 판정), quote: 판정 근거로 LLM이 인용한 고객 발화 일부.
        코드가 대조하는 것: 질문이 이전 턴에 걸렸나 / 인용이 이번 고객 발화에 실제로 있나.
        unclear이거나 대조에 실패하면 질문을 그대로 둔다 (확정하지 않는 쪽이 안전).
        """
        p = self.pending
        verified = bool(p and p.turn < self.turn and answer in ("yes", "no")
                        and quote.strip() and quote.strip() in utterance)
        self.trace.add(self.turn, "pending_answer", {
            "kind": p.kind if p else None, "action": p.action if p else "", "answer": answer,
            "quote": quote, "verified": verified})
        if not verified:
            return False
        if answer == "yes" and p.kind == "confirm":
            self.confirmations.append(Confirmation(p.action, p.target, self.turn))
        elif answer == "yes" and p.kind == "contact":
            self.contact_verified = True
        self.pending = None
        return True

    def is_confirmed(self, action: str, target: dict[str, Any]) -> bool:
        """같은 도구·같은 대상으로 재확인된 기록이 있는가. 실행 대상의 키 값이 하나라도 다르면 무효."""
        return any(c.action == action and covers(c.target, target) for c in self.confirmations)

    def unexecuted_confirmations(self) -> list[Confirmation]:
        """고객이 동의했는데 아직 실행이 성공하지 않은 것 (안 했거나 실패)."""
        return [c for c in self.confirmations
                if not any(a == c.action and covers(c.target, t) for a, t in self.succeeded)]
