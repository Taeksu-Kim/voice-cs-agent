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
    assert hist == [["customer", "택배 어디쯤이에요"], ["agent", "배송중이에요."]]


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
