"""업무 프로토콜(매뉴얼) — 업무 절차를 단계로 나눠 YAML로 정의한다 (protocols/*.yaml).

지금 어느 단계인지는 **코드가 아는 사실로만** 정한다 — 조회했나, 인증됐나·몇 번 실패했나,
확인 질문이 걸려 있나, 실행이 성공했나. LLM이 slots에 적은 값으로 정하지 않는다
(slots는 모델이 적을 때도 있고 안 적을 때도 있다 — 2026-10-13 s1_b2).

단계마다 지시(say)와 쓸 수 있는 도구(tools)가 다르다. 위에서부터 when 조건이 모두 맞는 첫 단계가 지금 단계.
when 어휘 (모두 AND):
  tool_done: 도구명          이 업무에서 그 도구가 성공했다 (not_tool_done: 반대)
  authenticated: true|false  본인 인증 상태
  auth_failures_gte: n       인증 실패 횟수
  pending: confirm|none      걸어 둔 확인 질문
  executed_any: [도구, ...]  그중 하나라도 실행에 성공했다 (not_executed_any: 반대)
  tool_empty: 도구명         그 조회 도구의 마지막 결과 목록이 전부 비었다 (예: lookup_cards 0장)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from csagent.agent.state import CallState

PROTOCOL_DIR = Path(__file__).parent / "protocols"
# 지시문 후보 실험용 (APO): 이 폴더에 같은 이름의 yaml이 있으면 그것을 읽는다. 저장소 지시문은 그대로 두고 후보만 끼워 돌린다
OVERRIDE_ENV = "CSAGENT_PROTOCOL_OVERRIDE"


@dataclass(frozen=True)
class Step:
    id: str
    say: str
    tools: tuple[str, ...] = ()
    when: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Protocol:
    name: str
    common: str                       # 모든 단계에 공통인 지시 (예외 상황 처리 등)
    steps: tuple[Step, ...]
    always_tools: tuple[str, ...] = ()   # 어느 단계에서나 쓸 수 있는 도구 (연결·종료)

    def current(self, state: CallState) -> Step:
        for step in self.steps:
            if all(_holds(state, k, v) for k, v in step.when.items()):
                return step
        return self.steps[-1]

    def tools(self, step: Step) -> tuple[str, ...]:
        return tuple(dict.fromkeys(step.tools + self.always_tools))

    def full_text(self) -> str:
        """전체 절차 (판정기의 근거 자료 등 — 에이전트 프롬프트에는 지금 단계만 넣는다)."""
        return self.common.rstrip() + "".join(f"\n\n[단계 {s.id}]\n{s.say.rstrip()}" for s in self.steps)

    def task_text(self, step: Step) -> str:
        return f"{self.common.rstrip()}\n\n{step_text(self, step)}"


def step_text(protocol: Protocol, step: Step) -> str:
    return f"[지금 단계: {step.id}]\n{step.say.rstrip()}\n이 단계에서 쓸 수 있는 도구: {', '.join(protocol.tools(step))}"


def _holds(state: CallState, key: str, value: Any) -> bool:
    done = state.active.tools_done if state.active is not None else set()
    executed = {a for a, _ in state.succeeded}
    if key == "tool_done":
        return value in done
    if key == "not_tool_done":
        return value not in done
    if key == "authenticated":
        return state.authenticated == bool(value)
    if key == "auth_failures_gte":
        return state.auth_failures >= int(value)
    if key == "pending":
        kind = state.pending.kind if state.pending is not None else "none"
        return kind == value
    if key == "executed_any":
        return bool(executed & set(value))
    if key == "not_executed_any":
        return not executed & set(value)
    if key == "tool_empty":
        return state.active is not None and value in state.active.tools_empty
    raise ValueError(f"알 수 없는 프로토콜 조건: {key}")


def protocol_path(name: str, directory: Optional[Path] = None) -> Path:
    if directory is None and os.environ.get(OVERRIDE_ENV):
        candidate = Path(os.environ[OVERRIDE_ENV]) / f"{name}.yaml"
        if candidate.exists():
            return candidate
    return (directory or PROTOCOL_DIR) / f"{name}.yaml"


def load_protocol(name: str, directory: Optional[Path] = None) -> Protocol:
    raw = yaml.safe_load(protocol_path(name, directory).read_text(encoding="utf-8"))
    steps = tuple(Step(s["id"], s["say"], tuple(s.get("tools") or ()), dict(s.get("when") or {}))
                  for s in raw["steps"])
    for s in steps:   # 오타 조건은 실행 중이 아니라 읽을 때 잡는다
        for k in s.when:
            if k not in {"tool_done", "not_tool_done", "authenticated", "auth_failures_gte", "pending",
                         "executed_any", "not_executed_any", "tool_empty"}:
                raise ValueError(f"{name}.yaml {s.id}: 알 수 없는 조건 {k}")
    return Protocol(raw["name"], raw["common"], steps, tuple(raw.get("always_tools") or ()))
