"""도구 계층 — 도구마다 위험도와 실행 조건을 선언하고, 실행은 레지스트리를 통해서만 한다.

위험도
- read     조회. 외부 영향 없음
- notify   링크 발송·재통화 예약·기록·확인 질문. 외부 영향은 있지만 되돌릴 수 있음
- execute  카드 정지·분실신고 등 되돌리기 어려운 실행 → 인증·재확인 대상
- handoff  상담원 연결·통화 종료

실행 조건(requires_*)은 「선언」일 뿐이고, 실제 검사는 실행 게이트(guardrails/gate.py)가 한다.
게이트를 끈 구성(A0)과 켠 구성(A1·C')을 같은 도구로 비교하기 위해서다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from csagent.agent.state import CallState

Risk = Literal["read", "notify", "execute", "handoff"]
ToolFn = Callable[..., dict]


@dataclass
class ToolSpec:
    name: str
    description: str
    params: dict[str, str] = field(default_factory=dict)   # 인자 이름 → 설명
    risk: Risk = "read"
    requires_auth: bool = False
    requires_confirmation: bool = False
    requires_contact: bool = False                          # 아웃바운드: 담당자 확인 후에만
    target_keys: tuple[str, ...] = ()                       # 재확인·중복 판정에 쓰는 인자
    asks_customer: bool = False                             # 고객에게 질문을 거는 도구 → 이 스텝의 말로 턴을 끝낸다
    fn: Optional[ToolFn] = None

    def target(self, args: dict[str, Any]) -> dict[str, Any]:
        return {k: args.get(k) for k in self.target_keys}

    def prompt_line(self) -> str:
        args = ", ".join(self.params)
        conds = [c for c, on in (("본인 인증 필요", self.requires_auth),
                                 ("고객 재확인 필요", self.requires_confirmation),
                                 ("담당자 확인 필요", self.requires_contact)) if on]
        cond = f" [{', '.join(conds)}]" if conds else ""
        return f"- {self.name}({args}): {self.description}{cond}"


@dataclass
class ToolOutcome:
    ok: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for s in specs:
            self.register(s)

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return list(self._specs)

    def prompt(self, names: Optional[list[str]] = None) -> str:
        return "\n".join(self._specs[n].prompt_line() for n in (names or self.names()) if n in self._specs)

    def run(self, state: CallState, name: str, args: dict[str, Any]) -> ToolOutcome:
        """도구 실행 — 인자 오류·백엔드 오류를 예외 대신 실패 결과로 돌려준다."""
        spec = self._specs.get(name)
        if spec is None or spec.fn is None:
            return ToolOutcome(False, error=f"unknown_tool:{name}")
        unknown = set(args) - set(spec.params)
        if unknown:
            return ToolOutcome(False, error=f"bad_args:{sorted(unknown)}")
        try:
            result = spec.fn(state, **args)
        except Exception as e:  # 백엔드 장애·시간 초과도 여기로
            return ToolOutcome(False, error=f"{type(e).__name__}: {e}")
        if isinstance(result, dict) and result.get("error"):
            return ToolOutcome(False, result=result, error=str(result["error"]))
        if spec.risk == "execute":
            state.executed.add(execution_key(spec, args))
            state.succeeded.append((spec.name, spec.target(args)))
        return ToolOutcome(True, result=result)


def execution_key(spec: ToolSpec, args: dict[str, Any]) -> str:
    return f"{spec.name}:{sorted(spec.target(args).items())}"
