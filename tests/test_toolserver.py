import shutil
from pathlib import Path

import anyio
import httpx
import pytest
from mcp import Client

from csagent.domain.app import create_app
from csagent.domain.store import FixtureStore
from csagent.toolserver.catalog import CatalogError, load_catalog
from csagent.toolserver.server import DEFAULT_CATALOG, ToolServer

FIXTURE = {"today": "2026-10-12", "auth_result": "success",
           "cards": [{"card_id": "C1", "name": "토스유스카드", "last4": "1234"}]}
CTX = {"csagent/context": {"customer_id": "s5:0:run"}}


def _server(catalog_dir=DEFAULT_CATALOG, store=None):
    store = store or FixtureStore()
    store.put("s5:0:run", FIXTURE)
    return ToolServer(catalog_dir, "http://domain", transport=httpx.ASGITransport(app=create_app(store)))


def _run(fn):
    return anyio.run(fn)


def test_list_tools_hides_context_params_and_carries_meta():
    async def main():
        async with Client(_server().server) as c:
            return {t.name: t for t in (await c.list_tools()).tools}
    tools = _run(main)
    assert set(tools) == {"lookup_orders", "lookup_transfers", "lookup_cards", "suspend_card", "report_card_lost",
                          "lookup_seller", "record_lead", "request_app_auth", "send_app_link", "schedule_callback"}
    suspend = tools["suspend_card"]
    assert list(suspend.input_schema["properties"]) == ["card_id"]          # customer_id는 모델에 안 보임
    assert suspend.input_schema["required"] == ["card_id"]
    assert suspend.meta["csagent/risk"] == "execute" and suspend.meta["csagent/requires_auth"] is True
    assert suspend.meta["csagent/context_params"] == ["customer_id"]
    assert suspend.meta["csagent/grounded_by"] == {"card_id": "lookup_cards.cards.card_id"}
    assert suspend.annotations.destructive_hint is True
    assert tools["request_app_auth"].meta["csagent/effects"] == ["authenticate"]


def test_call_tool_routes_to_domain_with_context_from_meta():
    async def main():
        async with Client(_server().server) as c:
            ok = await c.call_tool("suspend_card", {"card_id": "C1"}, meta=CTX)
            again = await c.call_tool("lookup_cards", {}, meta=CTX)
            no_ctx = await c.call_tool("lookup_cards", {})
            forged = await c.call_tool("lookup_cards", {"customer_id": "other"}, meta=CTX)
            return ok, again, no_ctx, forged
    ok, again, no_ctx, forged = _run(main)
    assert ok.is_error is False and ok.structured_content == {"ok": True, "data": {"status": "suspended", "card_id": "C1"}}
    assert "csagent/domain_ms" in ok.meta
    assert again.structured_content["data"]["cards"][0]["status"] == "일시정지"
    assert no_ctx.is_error and no_ctx.structured_content["error"]["code"] == "CONTEXT_MISSING"
    assert forged.is_error and forged.structured_content["error"]["code"] == "INVALID_ARGS"


def test_domain_errors_and_backend_down_keep_one_shape():
    async def main():
        async with Client(_server().server) as c:
            unknown = await c.call_tool("suspend_card", {"card_id": "9"}, meta=CTX)
        down = ToolServer(DEFAULT_CATALOG, "http://127.0.0.1:9", timeout_s=0.5)
        async with Client(down.server) as c:
            gone = await c.call_tool("lookup_cards", {}, meta=CTX)
        return unknown, gone
    unknown, gone = _run(main)
    assert unknown.structured_content["error"]["code"] == "CARD_NOT_FOUND"
    assert gone.is_error and gone.structured_content["error"]["code"] in {"BACKEND_UNAVAILABLE", "BACKEND_TIMEOUT"}


def test_bad_catalog_rejected_and_reload_keeps_previous(tmp_path: Path):
    for f in Path(DEFAULT_CATALOG).glob("*.yaml"):
        shutil.copy(f, tmp_path / f.name)
    srv = _server(tmp_path)
    before = srv.catalog.digest
    card = tmp_path / "card.yaml"
    card.write_text(card.read_text(encoding="utf-8").replace("risk: execute", "risk: dangerous", 1), encoding="utf-8")
    with pytest.raises(CatalogError, match="risk"):
        srv.reload()
    assert srv.catalog.digest == before
    (tmp_path / "card.yaml").write_text("- name: lookup_orders\n  owner: x\n  description: d\n  endpoint: /x\n  risk: read\n",
                                        encoding="utf-8")
    with pytest.raises(CatalogError, match="중복"):
        load_catalog(tmp_path)
