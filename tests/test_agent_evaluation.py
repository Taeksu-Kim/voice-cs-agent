import json
from pathlib import Path

from csagent.agent.baseline import BaselineAgent
from csagent.agent.contracts import CallContext, TurnInput
from csagent.agent.guardrails.gate import ExecutionGate
from csagent.agent.interface import new_call
from csagent.agent.tools.mock import default_registry
from csagent.evaluation.runner import run_scenario
from csagent.evaluation.scenario import Expectation, Scenario, load_scenarios
from csagent.evaluation.scoring import LLMJudge, score
from csagent.evaluation.simulator import ScriptedCustomer
from csagent.llm import ScriptedLLM, parse_json
from csagent.retrieval.models import Hit


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, k=5, *, flt=None, **_):
        self.calls.append((query, flt))
        return [Hit("faq:4705#0", 1.0, "faq:4705", "faq", "배송 기간", "", "평균 3-5일\n[앱 링크: 주문 내역]")]


def _say(text, tool=None):
    return json.dumps({"say": text, "tool": tool}, ensure_ascii=False)


CTX = CallContext(audience="consumer", today="2026-10-12",
                  customer={"name": "김민지", "orders": [{"order_id": "A1", "status": "배송중"}]})


def make_agent(llm, retriever=None, gate=True):
    return BaselineAgent(retriever or FakeRetriever(), llm, default_registry(),
                         gate=ExecutionGate() if gate else None)


def ask(agent, text, state=None):
    state = state or new_call(CTX)
    return agent.respond(state, TurnInput(text)), state


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"say": "네"}\n```') == {"say": "네"}


def test_agent_lookup_tool_then_answer():
    llm = ScriptedLLM([_say("", {"name": "lookup_orders", "args": {}}), _say("배송중이에요.")])
    retriever = FakeRetriever()
    reply, _ = ask(make_agent(llm, retriever), "내 택배 어디쯤이에요")
    assert reply.text == "배송중이에요."
    assert [c.name for c in reply.tool_calls] == ["lookup_orders"]
    assert reply.tool_calls[0].result["ok"] and reply.tool_calls[0].result["orders"][0]["order_id"] == "A1"
    assert "[도구 결과]" in llm.calls[1][-1]["content"]          # 조회 결과를 다시 넣고 답했다
    assert retriever.calls[0][1].audience == "consumer"         # 통화 맥락으로 필터


def test_agent_action_tool_sets_disposition():
    llm = ScriptedLLM([_say("상담원 연결해 드릴게요.", {"name": "transfer_to_agent", "args": {"reason": "지연"}})])
    reply, _ = ask(make_agent(llm), "상담원 바꿔주세요")
    assert reply.end_call and reply.disposition == "transferred"


def test_agent_falls_back_on_bad_json():
    reply, _ = ask(make_agent(ScriptedLLM(["말도 안 되는 출력"])), "어")
    assert "다시 말씀" in reply.text


def _scenario(**expect):
    return Scenario("t1", "S1", "B0", CTX, {"goal": "배송 기간"}, Expectation(**expect), max_turns=4)


def test_runner_and_rule_scoring():
    llm = ScriptedLLM([_say("평균 삼에서 오 영업일 걸려요."),
                       _say("감사합니다. 좋은 하루 보내세요.", {"name": "end_call", "args": {"disposition": "resolved"}})])
    tr = run_scenario(make_agent(llm), ScriptedCustomer(["배송 며칠 걸려요", "네 감사합니다"]),
                      _scenario(dispositions=["resolved"], tools_forbidden=["lookup_orders"],
                                gold_docs=["faq:4705"]))
    assert tr.disposition == "resolved" and tr.end_reason == "agent_ended"
    sco = score(_scenario(dispositions=["resolved"], tools_forbidden=["lookup_orders"],
                          gold_docs=["faq:4705"]), tr)
    assert sco.passed, [c.__dict__ for c in sco.failed()]


def test_voice_format_check_flags_lists_and_parens():
    llm = ScriptedLLM([_say("고객센터(1644-2944)로\n- 문의하세요")])
    tr = run_scenario(make_agent(llm), ScriptedCustomer(["지연돼요"]), _scenario())
    names = {c.name for c in score(_scenario(), tr).failed()}
    assert "voice_format" in names


def test_customer_hangup_without_satisfaction_is_unresolved():
    llm = ScriptedLLM([_say("확인해 볼게요.")])
    tr = run_scenario(make_agent(llm), ScriptedCustomer(["택배요"], satisfied=False), _scenario())
    assert tr.disposition == "unresolved"


def test_judge_yes_no_to_pass_fail():
    sc = _scenario(must_mention=["평균 3~5일"], must_not=["도착일 확정"])
    llm = ScriptedLLM([_say("평균 삼에서 오 일이요.")])
    tr = run_scenario(make_agent(llm), ScriptedCustomer(["며칠요"], satisfied=True), sc)
    judge = LLMJudge(ScriptedLLM([json.dumps({"answers": [
        {"id": "mention_0", "answer": "YES", "reason": "말함"},
        {"id": "must_not_0", "answer": "NO", "reason": "안 함"}]})]))
    checks = {c.name: c.passed for c in judge.checks(sc, tr)}
    assert checks == {"mention_0": True, "must_not_0": True}


def test_scenario_files_load():
    scs = load_scenarios(Path(__file__).resolve().parent.parent / "scenarios")
    assert len(scs) >= 5 and all(s.expect.dispositions for s in scs)


def test_say_extractor_streams_only_say_value():
    from csagent.agent.streaming import SayExtractor
    raw = '{"say": "배송은 \\"평균\\" 3일\\n이에요 \\uAC00", "tool": null}'
    ex, out = SayExtractor(), ""
    for i in range(0, len(raw), 2):  # 2글자씩 쪼개 들어와도
        out += ex.feed(raw[i:i + 2])
    assert out == '배송은 "평균" 3일\n이에요 가' and ex.done


def test_respond_stream_event_order():
    llm = ScriptedLLM([_say("", {"name": "lookup_orders", "args": {}}), _say("배송중이에요.")])
    events = list(make_agent(llm).respond_stream(new_call(CTX), TurnInput("택배요")))
    types = [e.type for e in events]
    assert types[0] == "retrieval" and types[-1] == "done"
    assert types.index("tool_call") < types.index("tool_result") < types.index("delta")
    assert "".join(e.data["text"] for e in events if e.type == "delta") == "배송중이에요."
