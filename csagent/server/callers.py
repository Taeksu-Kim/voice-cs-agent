"""실제 테스트 탭의 발신자(고객) — scenarios_live/*.yaml. 평가(run_eval, scenarios/)에는 들어가지 않는다.

파일 하나 = 고객 한 명: 가짜 CRM(context.customer) + 상황 카드(situations).
상황 카드는 테스터가 고르는 설명뿐이다 — 대사·기대 결과가 없고 테스터가 설명을 보고 알아서 말한다.
related는 그 상황에서 테스터가 말할 때 필요한 CRM 값의 참조 (order:A3002 · card:C1 · transfer:T1 · seller).
참조는 로드할 때 CRM에 실제 있는지 확인한다 — 화면에 보인 번호는 에이전트가 조회할 수 있는 값이어야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from csagent.agent.contracts import CallContext
from csagent.evaluation.scenario import Expectation, Scenario

# 참조 종류 → (CRM 목록 키, 항목 id 키). seller는 목록이 아니라 한 덩어리
_REFS = {"order": ("orders", "order_id"), "card": ("cards", "card_id"), "transfer": ("transfers", "transfer_id")}


@dataclass
class Situation:
    id: str
    title: str
    desc: str
    related: list[str] = field(default_factory=list)


@dataclass
class Caller:
    id: str
    name: str
    profile: str               # 목록에 보일 고객 설명 (예: 토스 앱 고객 · 주문 3 · 카드 2 · 송금 2)
    context: CallContext
    situations: list[Situation]

    def to_scenario(self) -> Scenario:
        """세션 생성·녹화 재생은 시나리오 단위로 돌아간다 — 발신자를 task=LIVE 시나리오로 끼워 넣는다."""
        return Scenario(id=self.id, task="LIVE", behavior="-", context=self.context,
                        customer={"goal": self.profile}, expect=Expectation())

    def view(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "profile": self.profile,
                "direction": self.context.direction, "audience": self.context.audience,
                "today": self.context.today, "customer": self.context.customer,
                "situations": [vars(s) for s in self.situations]}


def _check_ref(customer: dict, ref: str, where: str) -> None:
    kind, _, key = ref.partition(":")
    if kind == "seller" and not key:
        if not customer.get("seller"):
            raise ValueError(f"{where}: CRM에 seller가 없음")
        return
    if kind not in _REFS:
        raise ValueError(f"{where}: 알 수 없는 참조 {ref} (order:· card:· transfer:· seller)")
    items, id_key = _REFS[kind]
    if not any(str(x.get(id_key)) == key for x in customer.get(items, [])):
        raise ValueError(f"{where}: CRM {items}에 {key} 없음")


def load_callers(directory: Path) -> list[Caller]:
    import yaml  # lazy

    out = []
    for path in sorted(Path(directory).glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        ctx = raw.get("context", {})
        customer = ctx.get("customer", {})
        situations = [Situation(id=s["id"], title=s["title"], desc=s["desc"], related=s.get("related", []))
                      for s in raw.get("situations", [])]
        for s in situations:
            for ref in s.related:
                _check_ref(customer, ref, f"{path.name} {s.id}")
        out.append(Caller(
            id=raw["id"], name=raw["name"], profile=raw["profile"],
            context=CallContext(direction=ctx.get("direction", "inbound"),
                                audience=ctx.get("audience", "consumer"),
                                companies=tuple(ctx.get("companies", [])),
                                service=ctx.get("service", "토스"),
                                customer=customer, today=str(ctx.get("today", ""))),
            situations=situations))
    return out
