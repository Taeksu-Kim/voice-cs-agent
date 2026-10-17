"""시나리오 = 업무 × 고객 행동 × 기대 결과. YAML로 선언적으로 정의한다 (scenarios/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from csagent.agent.contracts import CallContext


@dataclass
class Expectation:
    dispositions: list[str] = field(default_factory=list)     # 허용되는 통화 결과
    must_mention: list[str] = field(default_factory=list)     # 고객에게 전달해야 할 내용 (Judge 채점)
    must_not: list[str] = field(default_factory=list)         # 하면 안 되는 말·행동 (Judge 채점)
    tools_required: list[str] = field(default_factory=list)   # 반드시 호출해야 할 도구
    required_by_turn: Optional[int] = None                    # tools_required를 이 턴까지 호출해야 함 (답하기 전 조회)
    tools_forbidden: list[str] = field(default_factory=list)  # 호출하면 안 되는 도구
    gold_docs: list[str] = field(default_factory=list)        # 검색되어야 할 문서(doc_id) 중 하나
    milestones: list[Any] = field(default_factory=list)       # 순서대로 거칠 지점 (evaluation/path.py 표기)


@dataclass
class Scenario:
    id: str
    task: str                 # 업무 코드 (S1 …)
    behavior: str             # 고객 행동 코드 (B0 …)
    context: CallContext
    customer: dict[str, Any]  # goal, persona, facts, opening, behavior_note
    expect: Expectation
    max_turns: int = 8


def load_scenarios(directory: Path) -> list[Scenario]:
    import yaml  # lazy

    out = []
    for path in sorted(Path(directory).glob("*.yaml")):
        for raw in yaml.safe_load(path.read_text(encoding="utf-8")):
            ctx = raw.get("context", {})
            out.append(Scenario(
                id=raw["id"], task=raw["task"], behavior=raw["behavior"],
                context=CallContext(
                    direction=ctx.get("direction", "inbound"),
                    audience=ctx.get("audience", "consumer"),
                    companies=tuple(ctx.get("companies", [])),
                    service=ctx.get("service", "토스"),
                    customer=ctx.get("customer", {}),
                    today=str(ctx.get("today", "")),
                ),
                customer=raw["customer"],
                expect=Expectation(**raw.get("expect", {})),
                max_turns=raw.get("max_turns", 8),
            ))
    return out
