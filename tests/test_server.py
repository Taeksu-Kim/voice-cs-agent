import json

from fastapi.testclient import TestClient

from csagent.agent.contracts import CallContext
from csagent.evaluation.scenario import Expectation, Scenario
from csagent.llm import ScriptedLLM
from csagent.server.app import create_app
from tests.test_agent_evaluation import _say, make_agent

SC = Scenario("s1_demo", "S1", "B0",
              CallContext(audience="consumer", today="2026-10-12",
                          customer={"name": "김민지", "orders": [{"order_id": "A1", "status": "배송중"}]}),
              {"goal": "배송 확인", "opening": "택배 어디쯤이에요"}, Expectation(), max_turns=4)


def _client(outputs):
    agent = make_agent(ScriptedLLM(outputs))
    return TestClient(create_app(agent, [SC], web_dir=None))


def _events(resp):
    out = []
    for block in resp.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_session_turn_streams_sse_and_records_history():
    c = _client([_say("", {"name": "lookup_orders", "args": {}}), _say("배송중이에요.")])
    assert c.get("/api/scenarios").json()[0]["customer_name"] == "김민지"
    sid = c.post("/api/sessions", json={"scenario_id": "s1_demo"}).json()["session_id"]

    resp = c.post(f"/api/sessions/{sid}/turn", json={"utterance": "택배 어디쯤이에요"})
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _events(resp)
    types = [t for t, _ in events]
    agent_types = [t for t in types if t not in ("trace", "step")]
    assert agent_types[0] == "retrieval" and types[-1] == "done" and "tool_call" in types
    # 실시간 관찰: 추적 기록이 같이 흘러나오고, 끝에 통화 상태가 붙는다
    kinds = [d["kind"] for t, d in events if t == "trace"]
    assert {"retrieval", "llm", "tool_request", "tool_result"} <= set(kinds)
    assert events[-1][1]["state"]["turn"] == 1
    assert "".join(d["text"] for t, d in events if t == "delta") == "배송중이에요."

    hist = c.get(f"/api/sessions/{sid}").json()["history"]
    assert hist[0][0] == "agent" and "AI 상담원" in hist[0][1]   # 인바운드 고정 인사
    assert hist[1:] == [["customer", "택배 어디쯤이에요"], ["agent", "배송중이에요."]]


def test_ended_call_rejects_more_turns():
    c = _client([_say("감사합니다.", {"name": "end_call", "args": {"disposition": "resolved"}})])
    sid = c.post("/api/sessions", json={"scenario_id": "s1_demo"}).json()["session_id"]
    done = _events(c.post(f"/api/sessions/{sid}/turn", json={"utterance": "끊을게요"}))[-1][1]
    assert done["session_ended"] and done["disposition"] == "resolved"
    assert c.post(f"/api/sessions/{sid}/turn", json={"utterance": "여보세요"}).status_code == 409


def test_unknown_session_and_scenario():
    c = _client([])
    assert c.post("/api/sessions", json={"scenario_id": "없음"}).status_code == 404
    assert c.get("/api/sessions/nope").status_code == 404


def _caller_app(tmp_path, outputs=()):
    from csagent.server.callers import load_callers
    live = tmp_path / "live"
    live.mkdir()
    (live / "c.yaml").write_text("""
id: live_c
name: 김민지
profile: 토스 앱 고객
context:
  today: 2026-10-12
  customer: {name: 김민지, orders: [{order_id: A1, item: 의자, status: 배송중}]}
situations:
  - {id: stuck, title: 배송 멈춤, desc: 의자가 안 온다, related: ["order:A1"]}
""", encoding="utf-8")
    log_dir = tmp_path / "logs" / "d1"
    log_dir.mkdir(parents=True)
    agent = make_agent(ScriptedLLM(list(outputs)))
    return TestClient(create_app(agent, [SC], callers=load_callers(live), web_dir=None,
                                 log_dir=log_dir, root=tmp_path))


def test_caller_situation_session_is_recorded_and_can_become_a_demo(tmp_path):
    c = _caller_app(tmp_path, [_say("감사합니다.", {"name": "end_call", "args": {"disposition": "resolved"}})])
    callers = c.get("/api/callers").json()
    assert callers[0]["situations"][0]["related"] == ["order:A1"]
    assert {s["id"] for s in c.get("/api/scenarios").json()} == {"live_c", "s1_demo"}
    assert c.post("/api/sessions", json={"scenario_id": "live_c", "situation": "없음"}).status_code == 404

    sid = c.post("/api/sessions", json={"scenario_id": "live_c", "situation": "stuck"}).json()["session_id"]
    _events(c.post(f"/api/sessions/{sid}/turn", json={"utterance": "끊을게요"}))
    recs = c.get("/api/recordings").json()
    assert recs[0]["id"] == sid and recs[0]["situation"] == "stuck"

    # 데모 목록: 실제 기록만 올릴 수 있고, 같은 기록은 덮어쓴다
    assert c.post("/api/demos", json={"title": "x", "source": "logs/d1", "record_id": "없음"}).status_code == 404
    c.post("/api/demos", json={"title": "배송 멈춤", "source": "logs/d1", "record_id": sid})
    items = c.post("/api/demos", json={"title": "배송 멈춤 2", "source": "logs/d1", "record_id": sid}).json()
    assert [d["title"] for d in items] == ["배송 멈춤 2"] == [d["title"] for d in c.get("/api/demos").json()]
    assert c.delete("/api/demos", params={"source": "logs/d1", "record_id": sid}).json() == []


def test_caller_related_must_exist_in_crm(tmp_path):
    import pytest
    from csagent.server.callers import load_callers
    (tmp_path / "c.yaml").write_text("""
id: x
name: x
profile: x
context: {customer: {cards: [{card_id: C1}]}}
situations: [{id: s, title: t, desc: d, related: ["card:C9"]}]
""", encoding="utf-8")
    with pytest.raises(ValueError, match="C9"):
        load_callers(tmp_path)
