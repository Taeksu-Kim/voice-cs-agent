from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from csagent.agent.guardrails.contacts import SpeechScreen
from csagent.agent.guardrails.links import strip_links
from csagent.domain.app import create_app
from csagent.domain.guides import ITEMS, GuideBook
from csagent.domain.store import FixtureStore
from csagent.toolserver.catalog import CatalogError, load_catalog, to_tool

ROOT = Path(__file__).resolve().parent.parent


def test_catalog_enum_matches_guide_items_and_urls_come_from_knowledge_base():
    book = GuideBook.load()
    tool = to_tool(load_catalog(ROOT / "catalog").get("send_app_link"))
    assert tool.input_schema["properties"]["target"]["enum"] == list(book.by_id)   # 두 목록이 같아야 한다
    g = book.get("wrong_transfer_report")
    assert g.doc_id == "faq:139" and g.url == "https://support.toss.im/faq/139"
    assert all(u.startswith(("https://support.toss.im/", "https://shopping-docs.toss.im/")) for u in book.urls())


def test_unknown_doc_id_stops_loading(tmp_path: Path):
    items = tmp_path / "guides.yaml"
    items.write_text(yaml.safe_dump([{"id": "x", "name": "x", "audience": "consumer", "doc_id": "faq:없음"}]),
                     encoding="utf-8")
    with pytest.raises(ValueError, match="faq:없음"):
        GuideBook.load(items)


def test_bad_enum_rejected(tmp_path: Path):
    (tmp_path / "notify.yaml").write_text(yaml.safe_dump([{
        "name": "send_app_link", "owner": "notify", "description": "d", "endpoint": "/notify/app_link", "risk": "notify",
        "params": {"target": {"type": "string", "source": "model", "enum": ["a", "a"]}}}]), encoding="utf-8")
    with pytest.raises(CatalogError, match="enum"):
        load_catalog(tmp_path)


def test_domain_sends_registered_guide_only():
    store = FixtureStore()
    store.put("c1", {})
    c = TestClient(create_app(store, guides=GuideBook.load(ITEMS)))
    sent = c.post("/notify/app_link", json={"customer_id": "c1", "target": "seller_penalty_appeal"}).json()
    assert sent["data"] == {"sent": True, "target": "seller_penalty_appeal", "title": "셀러 페널티 소명 가이드",
                            "url": "https://shopping-docs.toss.im/penalty/appeal/seller-penalty"}
    bad = c.post("/notify/app_link", json={"customer_id": "c1", "target": "https://evil.example/x"}).json()
    assert bad["ok"] is False and bad["error"]["code"] == "UNKNOWN_GUIDE"


def test_speech_drops_urls_from_url_to_sentence_end():
    say = "문자로 보내 드렸어요. 주소는 support.toss.im/faq/139 이에요. 확인해 보세요."
    text, urls = strip_links(say)
    assert text == "문자로 보내 드렸어요. 주소는 확인해 보세요."
    assert urls == ["support.toss.im/faq/139"]
    assert strip_links("https://shopping-docs.toss.im/penalty 에서 보세요.")[0] == ""
    # URL이 아닌 영문·숫자는 그대로
    for keep in ("KB국민카드는 1.5% 적립돼요.", "토스 앱 iOS 버전에서 보세요.", "평균 3-5일 걸려요."):
        assert strip_links(keep) == (keep, [])
    # 스트리밍으로 나간 말도 기록과 같아야 한다 — 조각이 URL 가운데에서 끊겨도
    for pieces in (["문자로 보내 드렸어요. 주소는 support.", "toss.im/faq/139 이에요.", " 확인해 보세요."], [say]):
        screen = SpeechScreen(None)
        assert "".join(screen.feed(p) for p in pieces) + screen.flush() == text
    screen = SpeechScreen(set())
    assert screen.feed("KB국민카드는 1.5% 적립돼요.") + screen.flush() == "KB국민카드는 1.5% 적립돼요."
