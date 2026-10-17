import json

from csagent.llm import LLMResult, ScriptedLLM
from csagent.llm_log import LLMCallLog, LoggedLLM, log_context


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_chat_is_logged_with_call_id_and_context(tmp_path):
    log = LLMCallLog(tmp_path / "calls.jsonl")
    llm = LoggedLLM(ScriptedLLM(['{"say": "네"}']), log, "customer")
    with log_context(scenario="s1", repeat=0):
        res = llm.chat([{"role": "user", "content": "안녕"}], temperature=0.7)
    [rec] = _records(log.path)
    assert res.call_id and rec["call_id"] == res.call_id
    assert rec["source"] == "customer" and rec["scenario"] == "s1" and rec["repeat"] == 0
    assert rec["messages"] == [{"role": "user", "content": "안녕"}]
    assert rec["output"] == '{"say": "네"}' and rec["params"] == {"temperature": 0.7}


def test_stream_is_logged_once_after_result(tmp_path):
    log = LLMCallLog(tmp_path / "calls.jsonl")
    llm = LoggedLLM(ScriptedLLM(['{"say": "확인해 볼게요"}']), log, "agent").with_source("router")
    items = list(llm.stream_chat([{"role": "user", "content": "x"}], json_mode=True))
    result = items[-1]
    assert isinstance(result, LLMResult) and result.call_id
    [rec] = _records(log.path)
    assert rec["call_id"] == result.call_id and rec["source"] == "router"
    assert rec["output"] == '{"say": "확인해 볼게요"}' and "error" not in rec


def test_aborted_stream_keeps_partial_output(tmp_path):
    log = LLMCallLog(tmp_path / "calls.jsonl")
    gen = LoggedLLM(ScriptedLLM(["abcdefghi"]), log, "agent").stream_chat([])
    next(gen)
    gen.close()
    [rec] = _records(log.path)
    assert rec["error"] == "aborted" and rec["output"] == "abc"
