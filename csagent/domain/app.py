"""도메인 API (FastAPI) — 서비스는 소유 기준 경로로 나눈다 (docs/08 5-1). 요청 처리는 무상태:
요청마다 customer_id와 업무 인자를 받고, 세션을 조회하지 않는다.

    POST /shopping/orders/lookup     {customer_id, order_id?}
    POST /transfer/transfers/lookup  {customer_id}
    POST /card/cards/lookup          {customer_id}
    POST /card/suspend               {customer_id, card_id}
    POST /card/report_lost           {customer_id, card_id}
    POST /seller/lookup              {customer_id}
    POST /ads/leads                  {customer_id, need?, budget?, interest?, grade?}
    POST /auth/app_auth              {customer_id}
    POST /notify/app_link            {customer_id, target}   target = 안내 항목 id (domain/guides.yaml)
    POST /notify/callback            {customer_id, when}
    PUT · GET · DELETE /_fixtures/{customer_id}   평가·개발 전용 (enable_fixtures=True일 때만)
    GET  /health

응답: {"ok": true, "data": {...}} | {"ok": false, "error": {"code", "message"}}

- 인증 검증은 데모에서 생략한다 (docs/08 7-1) — requires_auth 도구도 항상 통과
- 결과·오류 문구는 옮기기 전 목업(agent/tools/mock.py, 커밋 3cbc761까지)과 같게 둔다 — 구조 변경 전후를 같은 조건으로 비교하기 위해서.
  「이미 정지된 카드」 같은 새 업무 거부는 이 단계에서 넣지 않는다
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from csagent.domain.guides import GuideBook
from csagent.domain.store import FixtureStore


class DomainError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def ok(data: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "data": data}


class Req(BaseModel):
    model_config = ConfigDict(extra="forbid")   # 모르는 인자는 INVALID_ARGS (에이전트 형식 검증의 뒷단)
    customer_id: str


class OrderReq(Req):
    order_id: str = ""


class CardReq(Req):
    card_id: str


class LeadReq(Req):
    need: str = ""
    budget: str = ""
    interest: str = ""
    grade: str = ""


class LinkReq(Req):
    target: str


class CallbackReq(Req):
    when: str


NO_TIME = {"", "unknown", "none", "null", "n/a", "?", "미정", "모름", "알 수 없음", "알수없음"}   # 「시각 없음」 표시


def create_app(store: Optional[FixtureStore] = None, enable_fixtures: bool = False,
               guides: Optional[GuideBook] = None) -> FastAPI:
    store = store if store is not None else FixtureStore()
    guides = guides if guides is not None else GuideBook.load()
    app = FastAPI(title="csagent domain API")
    app.state.store = store

    @app.exception_handler(DomainError)
    async def _domain_error(_: Request, e: DomainError) -> JSONResponse:
        return JSONResponse({"ok": False, "error": {"code": e.code, "message": e.message}}, status_code=e.status)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, e: RequestValidationError) -> JSONResponse:
        msg = "; ".join(f"{'.'.join(str(x) for x in err['loc'][1:])}: {err['msg']}" for err in e.errors())
        return JSONResponse({"ok": False, "error": {"code": "INVALID_ARGS", "message": msg}}, status_code=422)

    def customer(cid: str, op: str) -> dict[str, Any]:
        """고객 데이터 + 장애 주입. op는 작업 이름(= 카탈로그 도구 이름) — 시나리오 tool_failures의 키."""
        data = store.get(cid)
        if data is None:
            raise DomainError("CUSTOMER_NOT_FOUND", f"customer_not_found:{cid}", 404)
        failure = (data.get("tool_failures") or {}).get(op)
        if failure:
            raise DomainError("BACKEND_TIMEOUT", f"{op} 백엔드 응답 없음 ({failure})", 504)
        return data

    def find_card(data: dict[str, Any], card_id: str) -> dict[str, Any]:
        cards = data.get("cards") or []
        card = next((c for c in cards if c["card_id"] == card_id), None)
        if card is None:
            raise DomainError("CARD_NOT_FOUND",
                              f"unknown_card:{card_id} (조회된 card_id: {', '.join(c['card_id'] for c in cards)})", 404)
        return card

    # ------------------------------------------------------------ 쇼핑 주문

    shopping = APIRouter(prefix="/shopping", tags=["shopping"])

    @shopping.post("/orders/lookup")
    def lookup_orders(req: OrderReq) -> dict:
        """order_id가 있으면 그 건만. 본인 명의로 등록된 번호가 아니면 빈 목록 + not_found (다른 주문으로 대신 답하지 않게)."""
        data = customer(req.customer_id, "lookup_orders")
        orders = data.get("orders") or []
        key = "".join(str(req.order_id or "").split()).upper()
        if not key:
            return ok({"today": data.get("today", ""), "orders": orders})
        hit = [o for o in orders if str(o.get("order_id", "")).upper() == key]
        return ok({"today": data.get("today", ""), "orders": hit, **({} if hit else {"not_found": key})})

    # ------------------------------------------------------------ 송금

    transfer = APIRouter(prefix="/transfer", tags=["transfer"])

    @transfer.post("/transfers/lookup")
    def lookup_transfers(req: Req) -> dict:
        data = customer(req.customer_id, "lookup_transfers")
        return ok({"today": data.get("today", ""), "transfers": data.get("transfers") or []})

    # ------------------------------------------------------------ 카드

    card = APIRouter(prefix="/card", tags=["card"])

    @card.post("/cards/lookup")
    def lookup_cards(req: Req) -> dict:
        data = customer(req.customer_id, "lookup_cards")
        return ok({"cards": [{"card_id": c["card_id"], "name": c["name"], "last4": c.get("last4", ""),
                              "status": c.get("status", "정상")} for c in data.get("cards") or []]})

    @card.post("/suspend")
    def suspend_card(req: CardReq) -> dict:
        found = find_card(customer(req.customer_id, "suspend_card"), req.card_id)
        found["status"] = "일시정지"
        return ok({"status": "suspended", "card_id": req.card_id})

    @card.post("/report_lost")
    def report_card_lost(req: CardReq) -> dict:
        found = find_card(customer(req.customer_id, "report_card_lost"), req.card_id)
        found["status"] = "분실신고"
        return ok({"status": "reported_lost", "card_id": req.card_id})

    # ------------------------------------------------------------ 셀러 · 광고

    seller = APIRouter(prefix="/seller", tags=["seller"])

    @seller.post("/lookup")
    def lookup_seller(req: Req) -> dict:
        data = customer(req.customer_id, "lookup_seller")
        return ok({"seller": data.get("seller") or {}})

    ads = APIRouter(prefix="/ads", tags=["ads"])

    @ads.post("/leads")
    def record_lead(req: LeadReq) -> dict:
        customer(req.customer_id, "record_lead")
        return ok({"recorded": True, "lead": {"need": req.need, "budget": req.budget,
                                              "interest": req.interest, "grade": req.grade}})

    # ------------------------------------------------------------ 고객 인증

    auth = APIRouter(prefix="/auth", tags=["auth"])

    @auth.post("/app_auth")
    def request_app_auth(req: Req) -> dict:
        """앱 인증 결과는 시나리오가 정한다(auth_result). 실패 횟수는 인증 서비스가 고객별로 센다."""
        data = customer(req.customer_id, "request_app_auth")
        if data.get("auth_result", "success") == "success":
            return ok({"status": "success"})
        data["_auth_failures"] = data.get("_auth_failures", 0) + 1
        return ok({"status": "failed", "failures": data["_auth_failures"]})

    # ------------------------------------------------------------ 알림 · 예약

    notify = APIRouter(prefix="/notify", tags=["notify"])

    @notify.post("/app_link")
    def send_app_link(req: LinkReq) -> dict:
        customer(req.customer_id, "send_app_link")
        guide = guides.get(req.target)
        if guide is None:   # 카탈로그 enum으로 에이전트에서 이미 막힌다 — 여기는 뒷단
            raise DomainError("UNKNOWN_GUIDE", f"unknown_guide:{req.target} (등록된 안내 항목만 보낼 수 있음)")
        return ok({"sent": True, "target": guide.id, "title": guide.name, "url": guide.url})

    @notify.post("/callback")
    def schedule_callback(req: CallbackReq) -> dict:
        customer(req.customer_id, "schedule_callback")
        # 예약에는 시각이 있어야 한다 — 시각 없이 예약하고 고객에겐 「4시에 연락드릴게요」라고 한 오동작 (docs/09 G1).
        # 「시각 없음」 표시만 거른다. 시각 표현이 맞는지(「오후 4시 이후」)는 판단하지 않는다
        if req.when.strip().lower() in NO_TIME:
            raise DomainError("WHEN_REQUIRED", "when_required: 고객이 말한 희망 시각이 없어 예약할 수 없음 — "
                                               "고객에게 통화 가능한 시간을 묻고, 고객이 말한 시각으로 다시 예약하세요")
        return ok({"scheduled": True, "when": req.when})

    for r in (shopping, transfer, card, seller, ads, auth, notify):
        app.include_router(r)

    # ------------------------------------------------------------ 픽스처 (평가·개발 전용)

    if enable_fixtures:
        fixtures = APIRouter(prefix="/_fixtures", tags=["fixtures"])

        @fixtures.put("/{customer_id}")
        def put_fixture(customer_id: str, body: dict[str, Any]) -> dict:
            store.put(customer_id, body)
            return ok({"customer_id": customer_id})

        @fixtures.get("/{customer_id}")
        def get_fixture(customer_id: str) -> dict:
            snap = store.snapshot(customer_id)
            if snap is None:
                raise DomainError("CUSTOMER_NOT_FOUND", f"customer_not_found:{customer_id}", 404)
            return ok(snap)

        @fixtures.delete("/{customer_id}")
        def delete_fixture(customer_id: str) -> dict:
            return ok({"deleted": store.delete(customer_id)})

        app.include_router(fixtures)

    @app.get("/health")
    def health() -> dict:
        return ok({"service": "domain", "fixtures": enable_fixtures, "customers": len(store)})

    return app


def create_dev_app() -> FastAPI:
    """uvicorn factory — 평가·개발용 (픽스처 경로 켬)."""
    return create_app(enable_fixtures=True)
