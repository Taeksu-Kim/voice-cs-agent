from fastapi.testclient import TestClient

from csagent.domain.app import create_app
from csagent.domain.store import FixtureStore, split_customer

CUSTOMER = {
    "name": "이서연", "outbound_task": None, "auth_result": "success",
    "orders": [{"order_id": "A1001", "item": "무선 이어폰", "status": "배송중"}],
    "cards": [{"card_id": "C1", "name": "토스유스카드", "last4": "1234"},
              {"card_id": "C2", "name": "토스뱅크 체크카드", "last4": "5678"}],
}


def _client(customer=CUSTOMER, cid="s5:0:run"):
    _, fixture = split_customer(customer)
    store = FixtureStore()
    store.put(cid, {**fixture, "today": "2026-10-12"})
    return TestClient(create_app(store, enable_fixtures=True)), cid


def test_split_keeps_call_context_out_of_fixture():
    call_ctx, fixture = split_customer(CUSTOMER)
    assert set(call_ctx) == {"name", "outbound_task"}
    assert "name" not in fixture and fixture["cards"][0]["card_id"] == "C1"


def test_lookup_orders_by_id_and_not_found():
    c, cid = _client()
    hit = c.post("/shopping/orders/lookup", json={"customer_id": cid, "order_id": " a1001 "}).json()
    assert hit == {"ok": True, "data": {"today": "2026-10-12", "orders": CUSTOMER["orders"]}}
    miss = c.post("/shopping/orders/lookup", json={"customer_id": cid, "order_id": "B9"}).json()["data"]
    assert miss["orders"] == [] and miss["not_found"] == "B9"


def test_suspend_changes_state_seen_by_next_lookup():
    c, cid = _client()
    r = c.post("/card/suspend", json={"customer_id": cid, "card_id": "C1"}).json()
    assert r == {"ok": True, "data": {"status": "suspended", "card_id": "C1"}}
    cards = c.post("/card/cards/lookup", json={"customer_id": cid}).json()["data"]["cards"]
    assert cards[0]["status"] == "일시정지" and cards[1]["status"] == "정상"
    # 화면용 조회도 같은 데이터를 본다
    assert c.get(f"/_fixtures/{cid}").json()["data"]["cards"][0]["status"] == "일시정지"


def test_unknown_card_is_structured_error_with_same_message():
    c, cid = _client()
    r = c.post("/card/report_lost", json={"customer_id": cid, "card_id": "1234"})
    assert r.status_code == 404
    assert r.json()["error"] == {"code": "CARD_NOT_FOUND", "message": "unknown_card:1234 (조회된 card_id: C1, C2)"}


def test_injected_failure_and_auth_failures_counted_per_customer():
    c, cid = _client({**CUSTOMER, "auth_result": "fail", "tool_failures": {"report_card_lost": "timeout"}})
    r = c.post("/card/report_lost", json={"customer_id": cid, "card_id": "C1"})
    assert r.status_code == 504 and r.json()["error"]["code"] == "BACKEND_TIMEOUT"
    assert "report_card_lost 백엔드 응답 없음 (timeout)" == r.json()["error"]["message"]
    assert c.post("/auth/app_auth", json={"customer_id": cid}).json()["data"] == {"status": "failed", "failures": 1}
    assert c.post("/auth/app_auth", json={"customer_id": cid}).json()["data"]["failures"] == 2


def test_customers_are_isolated_and_unknown_args_rejected():
    c, cid = _client()
    c.post("/card/suspend", json={"customer_id": cid, "card_id": "C1"})
    _, fixture = split_customer(CUSTOMER)
    c.put("/_fixtures/s5:1:run", json=fixture)   # 같은 시나리오의 다음 반복
    cards = c.post("/card/cards/lookup", json={"customer_id": "s5:1:run"}).json()["data"]["cards"]
    assert cards[0]["status"] == "정상"
    r = c.post("/card/suspend", json={"customer_id": cid, "card_id": "C1", "pin": "0000"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_ARGS"
    assert c.post("/card/cards/lookup", json={"customer_id": "nobody"}).json()["error"]["code"] == "CUSTOMER_NOT_FOUND"


def test_fixture_routes_off_by_default():
    assert TestClient(create_app()).get("/_fixtures/x").status_code == 404
