"""콜 인프라 대역 — 통화를 열 때 고객을 정하고 도메인 API에 업무 데이터를 심는다 (docs/08 6절 1단계, 9절).

실서비스에서는 CTI가 발신번호로 고객을 찾아 상담 화면·세션을 여는 역할이다.
모델이 쓰는 도구가 아니므로 MCP를 거치지 않고 도메인 API의 /_fixtures를 직접 부른다.

- 시나리오 context.customer → 통화 컨텍스트(name·outbound_task, 에이전트에 남김) + 업무 데이터(도메인 API로)
- customer_id는 통화마다 고유 (cust-<call_id>) — 같은 시나리오를 반복해도 앞 통화의 변경이 새지 않고, 로그의 call_id로 바로 찾는다
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Protocol

import httpx

from csagent.agent.contracts import CallContext
from csagent.agent.interface import new_call
from csagent.agent.state import CallState
from csagent.domain.store import FixtureStore, split_customer


class Fixtures(Protocol):
    def put(self, customer_id: str, fixture: dict[str, Any]) -> None: ...
    def get(self, customer_id: str) -> Optional[dict[str, Any]]: ...
    def delete(self, customer_id: str) -> None: ...


class HttpFixtures:
    """도메인 API(평가·개발 모드)의 /_fixtures."""

    def __init__(self, domain_url: str, timeout_s: float = 5.0) -> None:
        self._http = httpx.Client(base_url=domain_url.rstrip("/"), timeout=timeout_s)

    def put(self, customer_id: str, fixture: dict[str, Any]) -> None:
        self._http.put(f"/_fixtures/{customer_id}", json=fixture).raise_for_status()

    def get(self, customer_id: str) -> Optional[dict[str, Any]]:
        r = self._http.get(f"/_fixtures/{customer_id}")
        return r.json()["data"] if r.status_code == 200 else None

    def delete(self, customer_id: str) -> None:
        self._http.delete(f"/_fixtures/{customer_id}")

    def health(self) -> dict[str, Any]:
        r = self._http.get("/health")
        r.raise_for_status()
        return r.json()["data"]


class StoreFixtures:
    """같은 프로세스의 목업 저장소 (테스트·로컬 스택)."""

    def __init__(self, store: FixtureStore) -> None:
        self.store = store

    def put(self, customer_id: str, fixture: dict[str, Any]) -> None:
        self.store.put(customer_id, fixture)

    def get(self, customer_id: str) -> Optional[dict[str, Any]]:
        return self.store.snapshot(customer_id)

    def delete(self, customer_id: str) -> None:
        self.store.delete(customer_id)


def open_call(ctx: CallContext, fixtures: Fixtures) -> CallState:
    """새 통화 — 에이전트는 통화 컨텍스트만 받고, 업무 데이터는 도메인 API에만 있다."""
    call_ctx, fixture = split_customer(ctx.customer)
    state = new_call(replace(ctx, customer=call_ctx))
    state.ctx.customer_id = f"cust-{state.call_id}"
    fixtures.put(state.ctx.customer_id, {**fixture, "today": ctx.today})
    state.trace.add(0, "call_open", {"customer_id": state.ctx.customer_id, "fixture": fixture})   # 재현용 초기 데이터
    return state


def close_call(state: CallState, fixtures: Fixtures) -> None:
    if state.ctx.customer_id:
        fixtures.delete(state.ctx.customer_id)
