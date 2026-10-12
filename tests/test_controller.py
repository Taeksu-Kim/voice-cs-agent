import json

from csagent.agent.contracts import CallContext, TurnInput
from csagent.agent.controller import CallController
from csagent.agent.guardrails.gate import ExecutionGate
from csagent.agent.interface import new_call
from csagent.agent.loop import TurnLoop
from csagent.agent.router import Router
from csagent.agent.tools.mock import default_registry
from csagent.agent.workers.specs import WORKERS
from csagent.llm import ScriptedLLM
from tests.test_agent_evaluation import FakeRetriever

CARDS = {"name": "이서연", "auth_result": "success",
         "cards": [{"card_id": "C1", "name": "토스유스카드", "last4": "1234"}]}


def _out(intent="continue", say="", tool=None, slots=None, answer=None):
    return json.dumps({"intent": intent, "pending_answer": answer, "slots": slots or {},
                       "say": say, "tool": tool}, ensure_ascii=False)


def _route(task):
    return json.dumps({"task": task, "reason": "테스트"})


def _controller(outputs):
    llm = ScriptedLLM(outputs)
    loop = TurnLoop(FakeRetriever(), llm, default_registry(), gate=ExecutionGate())
    return CallController(loop, Router(llm, WORKERS), WORKERS), llm


def test_routes_inbound_and_keeps_task_without_rerouting():
    c, llm = _controller([
        _route("card_loss"),
        _out(say="많이 놀라셨죠. 어떤 카드인가요?", slots={"lost": True}),
        _out(say="토스유스카드 맞으시죠?"),          # 2턴: 라우터 호출 없이 바로 워커
    ])
    s = new_call(CallContext(customer=CARDS))
    c.respond(s, TurnInput("카드를 잃어버렸어요"))
    assert s.active.name == "card_loss" and s.active.slots == {"lost": True}
    c.respond(s, TurnInput("유스카드요"))
    assert len(s.trace.of("route")) == 1             # 진행 중 업무가 있으면 라우터를 다시 부르지 않는다


def test_switch_hides_first_worker_speech_and_queues_previous_task():
    c, llm = _controller([
        _route("general"),
        _out(say="배송은 평균 삼에서 오 일이에요."),
        # 2턴: general 워커가 switch → 이 워커의 say는 고객에게 나가면 안 됨
        _out(intent="switch", say="이 말은 나가면 안 됨"),
        _route("card_loss"),
        _out(say="카드를 잃어버리셨군요. 확인해 볼게요."),
    ])
    s = new_call(CallContext(customer=CARDS))
    c.respond(s, TurnInput("배송 며칠 걸려요"))
    events = list(c.respond_stream(s, TurnInput("아 그리고 카드도 잃어버렸어요")))
    spoken = "".join(e.data["text"] for e in events if e.type == "delta")
    assert "나가면 안 됨" not in spoken and spoken == "카드를 잃어버리셨군요. 확인해 볼게요."
    assert s.active.name == "card_loss" and [t.name for t in s.queue] == ["general"]


def test_done_resumes_queued_task():
    c, _ = _controller([_route("card_loss"), _out(intent="done", say="처리됐어요.")])
    s = new_call(CallContext(customer=CARDS))
    from csagent.agent.state import TaskState
    s.queue.append(TaskState("general"))
    c.respond(s, TurnInput("카드 정지해 주세요"))
    assert s.active.name == "general"                 # 끝난 업무 다음엔 대기열의 업무로


def test_outbound_starts_preset_task_without_router():
    c, llm = _controller([_out(say="토스쇼핑 AI 상담원입니다. 담당자이신가요?",
                               tool={"name": "ask_contact", "args": {}}),
                          _out(say="담당자이신가요?")])
    s = new_call(CallContext(direction="outbound", audience="seller",
                             customer={"outbound_task": "penalty_notice", "seller": {"penalty_points_30d": 7}}))
    c.respond(s, TurnInput("여보세요"))
    assert s.active.name == "penalty_notice"
    assert s.trace.of("route")[0].data["reason"] == "아웃바운드 지정 업무"
    assert s.pending is not None and s.pending.kind == "contact"


def test_worker_cannot_use_tools_outside_its_policy():
    c, _ = _controller([
        _route("general"),
        _out(tool={"name": "report_card_lost", "args": {"card_id": "C1"}}),   # general 워커에는 없는 도구
        _out(say="그건 제가 바로 처리할 수 없어요."),
    ])
    s = new_call(CallContext(customer=CARDS))
    c.respond(s, TurnInput("카드 분실신고 해주세요"))
    assert s.trace.of("gate")[0].data["reason"] == "not_allowed"
