import json
import logging

from fastapi.testclient import TestClient

from csagent.llm import ScriptedLLM
from csagent.llm_log import LLMCallLog, LoggedLLM
from csagent.monitoring import aggregate, find_llm_call
from csagent.obs import CALLS_FILE, EVENTS_FILE, LLM_CALLS_FILE, event, log_context, setup_logging
from csagent.server.app import create_app
from tests.test_agent_evaluation import _say, make_agent
from tests.test_server import SC


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_event_is_json_line_with_context_and_fields(tmp_path):
    setup_logging(tmp_path)
    log = logging.getLogger("csagent.test")
    with log_context(scenario="s1", repeat=0):
        event(log, "fallback", logging.WARNING, call_id="c1", turn=2, reason="parse_error")
    [rec] = _lines(tmp_path / EVENTS_FILE)
    assert rec["event"] == "fallback" and rec["level"] == "WARNING"
    assert rec["scenario"] == "s1" and rec["call_id"] == "c1" and rec["turn"] == 2
    setup_logging(None)


def test_agent_turn_emits_call_start_turn_and_tool_events(tmp_path):
    setup_logging(tmp_path)
    from csagent.agent.contracts import TurnInput
    from csagent.agent.interface import new_call
    agent = make_agent(ScriptedLLM([_say("", {"name": "lookup_orders", "args": {}}), _say("배송중이에요.")]))
    state = new_call(SC.context)
    agent.respond(state, TurnInput("택배 어디쯤이에요"))
    recs = _lines(tmp_path / EVENTS_FILE)
    assert [r["event"] for r in recs] == ["call_start", "tool", "turn"]
    assert all(r["call_id"] == state.call_id for r in recs)
    turn = recs[-1]
    assert turn["tools"] == ["lookup_orders"] and turn["llm_steps"] == 2 and turn["chars"] == len("배송중이에요.")
    setup_logging(None)


def test_aggregate_reads_incrementally_and_skips_partial_line(tmp_path):
    ev = tmp_path / EVENTS_FILE
    ev.write_text(json.dumps({"ts": 1.0, "level": "INFO", "event": "turn", "ms_total": 1000, "chars": 200,
                              "ms_first_token": 300, "llm_steps": 2, "prompt_tokens": 10,
                              "completion_tokens": 2}) + "\n" + '{"ts": 2.0, "event": "fall', encoding="utf-8")
    m = aggregate(tmp_path, now=10.0)
    assert m["turns"]["n"] == 1 and m["turns"]["over_chars"] == 1 and m["incidents"]["fallback"] == {}
    with ev.open("a", encoding="utf-8") as f:   # 쓰던 줄이 마저 써짐
        f.write('back", "level": "WARNING", "reason": "no_say"}\n')
    m = aggregate(tmp_path, now=10.0)
    assert m["incidents"]["fallback"] == {"no_say": 1} and len(m["warnings"]) == 1


def test_llm_stats_and_raw_lookup(tmp_path):
    llm = LoggedLLM(ScriptedLLM(["a", "b"]), LLMCallLog(tmp_path / LLM_CALLS_FILE), "agent")
    r1 = llm.chat([{"role": "user", "content": "첫 질문"}])
    llm.with_source("judge").chat([{"role": "user", "content": "판정"}])
    m = aggregate(tmp_path)
    assert set(m["llm"]["by_source"]) == {"agent", "judge"} and m["llm"]["last_60s"]["calls"] == 2
    assert find_llm_call(tmp_path, r1.call_id)["messages"][0]["content"] == "첫 질문"
    assert find_llm_call(tmp_path, "없는id") is None


def test_server_saves_ended_call_and_serves_monitor(tmp_path):
    log_dir = tmp_path / "logs" / "20261013"
    setup_logging(log_dir)
    agent = make_agent(ScriptedLLM([_say("이용해 주셔서 감사해요.",
                                         {"name": "end_call", "args": {"disposition": "resolved"}})]))
    c = TestClient(create_app(agent, [SC], web_dir=None, log_dir=log_dir, root=tmp_path))
    sid = c.post("/api/sessions", json={"scenario_id": "s1_demo"}).json()["session_id"]
    c.post(f"/api/sessions/{sid}/turn", json={"utterance": "됐어요 끊을게요"})

    [saved] = _lines(log_dir / CALLS_FILE)
    assert saved["call_id"] == sid and saved["disposition"] == "resolved" and saved["trace"]
    m = c.get("/api/monitor").json()                      # 기본값 = 이 서버의 로그 폴더
    assert m["calls"] == {"started": 1, "ended": 1, "dispositions": {"resolved": 1}}
    assert c.get("/api/monitor/sources").json()["server"] == "logs/20261013"
    assert c.get("/api/monitor", params={"source": "logs/20261013"}).status_code == 200
    assert c.get("/api/monitor", params={"source": "../etc"}).status_code == 404
    setup_logging(None)
