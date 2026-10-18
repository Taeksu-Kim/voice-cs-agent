"""목업 저장소 — customer_id별 업무 데이터 (시나리오 픽스처).

평가 러너·웹 서버가 통화 시작 때 /_fixtures로 심고 끝나면 지운다 (docs/08 9절).
customer_id를 통화마다 고유하게 쓰므로 같은 시나리오를 반복해도 앞 통화의 변경(카드 정지 등)이 새지 않는다.

픽스처 필드 (시나리오 context.customer의 업무 데이터 부분)
- orders · transfers · cards · seller       업무 데이터
- today                                      시나리오 기준일 (조회 결과에 들어감)
- auth_result: "success" | "fail"            앱 인증 결과
- tool_failures: {"report_card_lost": "timeout"}   장애 주입 — 키는 작업 이름(= 카탈로그 도구 이름)
"""

from __future__ import annotations

import copy
import threading
from typing import Any, Optional

# 시나리오 context.customer 중 도메인 API로 가는 키. 나머지(name·outbound_task)는 통화 컨텍스트로 에이전트에 남는다
FIXTURE_KEYS = ("orders", "transfers", "cards", "seller", "auth_result", "tool_failures")


def split_customer(customer: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """시나리오 context.customer → (통화 컨텍스트, 업무 데이터 픽스처)."""
    fixture = {k: copy.deepcopy(v) for k, v in customer.items() if k in FIXTURE_KEYS}
    call_ctx = {k: v for k, v in customer.items() if k not in FIXTURE_KEYS}
    return call_ctx, fixture


class FixtureStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def put(self, customer_id: str, fixture: dict[str, Any]) -> None:
        with self._lock:
            self._data[customer_id] = copy.deepcopy(fixture)

    def get(self, customer_id: str) -> Optional[dict[str, Any]]:
        """저장된 데이터 자체를 돌려준다 (핸들러가 고쳐 쓴다). 화면 조회용은 snapshot."""
        with self._lock:
            return self._data.get(customer_id)

    def snapshot(self, customer_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            data = self._data.get(customer_id)
            return copy.deepcopy(data) if data is not None else None

    def delete(self, customer_id: str) -> bool:
        with self._lock:
            return self._data.pop(customer_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
