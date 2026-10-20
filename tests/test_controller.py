import json

from csagent.agent.contracts import CallContext, TurnInput
from csagent.agent.controller import CallController
from csagent.agent.guardrails.gate import ExecutionGate
from csagent.agent.loop import TurnLoop
from csagent.agent.router import Router
from tests.local import TOOLS, call, fixture, run_call
from csagent.agent.workers.specs import WORKERS
from csagent.llm import ScriptedLLM
from tests.test_agent_evaluation import FakeRetriever

CARDS = {"name": "이서연", "auth_result": "success",
         "cards": [{"card_id": "C1", "name": "토스유스카드", "last4": "1234"}]}


def _out(intent="continue", say="", tool=None, slots=None, answer=None):
    return json.dumps({"intent": intent, "pending_answer": answer, "slots": slots or {},
                       "tool": tool, "say": say}, ensure_ascii=False)   # tool이 say 앞 (실제 순서)


def _route(task):
    return json.dumps({"task": task, "reason": "테스트"})


def _controller(outputs):
    llm = ScriptedLLM(outputs)
    loop = TurnLoop(FakeRetriever(), llm, TOOLS, gate=ExecutionGate())
    return CallController(loop, Router(llm, WORKERS), WORKERS), llm


def test_routes_inbound_and_keeps_task_without_rerouting():
    c, llm = _controller([
        _route("card_loss"),
        _out(say="많이 놀라셨죠. 어떤 카드인가요?", slots={"lost": True}),
        _out(say="토스유스카드 맞으시죠?"),          # 2턴: 라우터 호출 없이 바로 워커
    ])
    s = call(CallContext(customer=CARDS))
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
        _out(say="카드를 잃어버리셨군요. 어느 카드인가요?"),
    ])
    s = call(CallContext(customer=CARDS))
    c.respond(s, TurnInput("배송 며칠 걸려요"))
    events = list(c.respond_stream(s, TurnInput("아 그리고 카드도 잃어버렸어요")))
    spoken = "".join(e.data["text"] for e in events if e.type == "delta")
    assert "나가면 안 됨" not in spoken and spoken == "카드를 잃어버리셨군요. 어느 카드인가요?"
    assert s.active.name == "card_loss" and [t.name for t in s.queue] == ["general"]


def test_done_resumes_queued_task():
    c, _ = _controller([_route("card_loss"), _out(intent="done", say="처리됐어요.")])
    s = call(CallContext(customer=CARDS))
    from csagent.agent.state import TaskState
    s.queue.append(TaskState("general"))
    c.respond(s, TurnInput("카드 정지해 주세요"))
    assert s.active.name == "general"                 # 끝난 업무 다음엔 대기열의 업무로


def test_outbound_starts_preset_task_without_router():
    c, llm = _controller([_out(say="토스쇼핑 AI 상담원입니다. 담당자이신가요?",
                               tool={"name": "ask_contact", "args": {}}),
                          _out(say="담당자이신가요?")])
    s = call(CallContext(direction="outbound", audience="seller",
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
    s = call(CallContext(customer=CARDS))
    c.respond(s, TurnInput("카드 분실신고 해주세요"))
    assert s.trace.of("gate")[0].data["reason"] == "not_allowed"


def test_two_requests_in_one_utterance_are_queued_in_order():
    c, _ = _controller([
        json.dumps({"tasks": ["card_loss", "wrong_transfer"], "reason": "두 용건"}),
        _out(say="카드부터 도와드릴게요."),
    ])
    s = call(CallContext(customer=CARDS))
    c.respond(s, TurnInput("카드 잃어버렸고 어제 송금도 잘못 보냈어요"))
    assert s.active.name == "card_loss" and [t.name for t in s.queue] == ["wrong_transfer"]


def test_add_intent_keeps_task_queues_new_one_and_end_call_waits():
    c, llm = _controller([
        _route("card_loss"),
        _out(say="어느 카드인가요?"),
        _out(intent="add", say="네, 송금 건도 이어서 도와드릴게요. 먼저 카드부터요."),
        json.dumps({"tasks": ["wrong_transfer"], "reason": "덧붙인 용건"}),
        # 3턴: 대기 용건이 있으니 라우터가 먼저 본다 (카드 얘기 → 전환 없음).
        # 카드 업무를 끝내며 통화를 끊으려 하면 → 미착수 용건 때문에 막힘
        _route("card_loss"),
        _out(intent="done", say="", tool={"name": "end_call", "args": {"disposition": "resolved"}}),
        _out(intent="done", say="카드는 처리됐어요. 이제 송금 건을 볼게요."),
    ])
    s = call(CallContext(customer=CARDS))
    c.respond(s, TurnInput("카드를 잃어버렸어요"))
    c.respond(s, TurnInput("유스카드요 아 그리고 어제 송금도 잘못했어요"))
    assert s.active.name == "card_loss" and [t.name for t in s.queue] == ["wrong_transfer"]
    reply = c.respond(s, TurnInput("네 됐어요"))
    assert not reply.end_call and s.active.name == "wrong_transfer"
    assert s.trace.of("gate")[-1].data["reason"] == "tasks_queued"


def test_add_survives_later_step_in_same_turn():
    # h4 실측: 확인 답 + 덧붙인 용건 → 1스텝 add+실행, 2스텝 결과 안내는 continue. add가 덮이면 대기열이 빈다
    c, _ = _controller([
        _route("card_loss"),
        _out(say="조회해 볼게요.", tool={"name": "lookup_cards", "args": {}}),
        _out(say="토스유스카드 맞으시죠?"),
        _out(intent="add", say="네, 송금 건도 이어서 볼게요.", tool={"name": "lookup_cards", "args": {}}),
        _out(intent="continue", say="카드부터 마저 볼게요."),
        json.dumps({"tasks": ["wrong_transfer"], "reason": "덧붙인 용건"}),
    ])
    s = call(CallContext(customer=CARDS))
    c.respond(s, TurnInput("카드를 잃어버렸어요"))
    c.respond(s, TurnInput("네 맞아요. 아 그리고 어제 송금도 잘못했어요"))
    assert [t.name for t in s.queue] == ["wrong_transfer"]


def test_end_call_block_is_once_then_allowed_next_turn():
    c, _ = _controller([
        _route("card_loss"),
        _out(intent="add", say="송금 건도 이어서 볼게요."),
        json.dumps({"tasks": ["wrong_transfer"], "reason": "덧붙인 용건"}),
        _route("card_loss"),
        # 2턴: 끝내려다 막힘 → 같은 턴 재시도도 막힘 → 남은 용건을 물어봄
        _out(say="", tool={"name": "end_call", "args": {"disposition": "resolved"}}),
        _out(say="", tool={"name": "end_call", "args": {"disposition": "resolved"}}),
        _out(say="송금 건도 이어서 도와드릴까요?"),
        # 3턴: 고객이 그래도 끝내겠다 → 허용
        _route("general"),
        _out(say="네, 이용해 주셔서 감사해요.", tool={"name": "end_call", "args": {"disposition": "resolved"}}),
    ])
    s = call(CallContext(customer=CARDS))
    c.respond(s, TurnInput("카드 잃어버렸고 송금도 잘못했어요"))
    reply = c.respond(s, TurnInput("이제 됐어요"))
    assert not reply.end_call and reply.text == "송금 건도 이어서 도와드릴까요?"
    assert [e.data["reason"] for e in s.trace.of("gate")] == ["tasks_queued", "tasks_queued"]
    reply = c.respond(s, TurnInput("아니요 그냥 끊을게요"))
    assert reply.end_call


def test_customer_raising_queued_task_switches_by_code():
    c, _ = _controller([
        _route("card_loss"),
        _out(intent="add", say="송금 건도 이어서 볼게요."),
        json.dumps({"tasks": ["wrong_transfer"], "reason": "덧붙인 용건"}),
        _route("wrong_transfer"),                     # 2턴: 고객이 대기 용건을 꺼냄 → 코드가 전환
        _out(say="확인해 볼게요.", tool={"name": "lookup_transfers", "args": {}}),
        _out(say="어느 송금인지 볼게요."),
    ])
    s = call(CallContext(customer={**CARDS, "transfers": []}))
    c.respond(s, TurnInput("카드 잃어버렸고 송금도 잘못했어요"))
    c.respond(s, TurnInput("어제 20만 원 잘못 보냈어요"))
    assert s.active.name == "wrong_transfer" and [t.name for t in s.queue] == ["card_loss"]
    assert s.trace.of("task_switch")[-1].data == {"from": "card_loss", "to": "wrong_transfer"}


def test_slot_rule_blocks_wrong_transfer_link_unless_mistake():
    c, _ = _controller([
        _route("wrong_transfer"),
        _out(say="확인해 볼게요.", tool={"name": "lookup_transfers", "args": {}}),
        _out(say="", slots={"transfer_reason": "거래대금"},
             tool={"name": "send_app_link", "args": {"target": "wrong_transfer_report"}}),
        _out(say="사기가 의심돼요. 상담원 연결해 드릴게요.",
             tool={"name": "transfer_to_agent", "args": {"reason": "거래 사기 의심", "summary": "중고거래"}}),
    ])
    s = call(CallContext(customer={"name": "윤가영", "transfers": []}))
    reply = c.respond(s, TurnInput("중고거래로 돈 보냈는데 판매자가 연락이 안 돼요"))
    assert [t.name for t in reply.tool_calls] == ["lookup_transfers", "send_app_link", "transfer_to_agent"]
    assert reply.tool_calls[1].result["blocked"] == "slot_rule"
    assert reply.disposition == "transferred"


def test_discarded_step_is_not_left_in_messages():
    c, llm = _controller([
        _route("wrong_transfer"),
        _out(say="사기는 아니죠?"),                  # 조회 전에 답하려 함 → 버림
        _out(say="확인해 볼게요.", tool={"name": "lookup_transfers", "args": {}}),
        _out(say="어떤 송금인지 볼게요."),
    ])
    s = call(CallContext(customer={"name": "윤가영", "transfers": []}))
    c.respond(s, TurnInput("돈 잘못 보냈어요"))
    retry = llm.calls[2]
    assert all("사기는 아니죠" not in m["content"] for m in retry if m["role"] == "assistant")   # 버린 응답은 대화에 안 넣는다
    assert retry[-1]["role"] == "user" and "[절차]" in retry[-1]["content"]


def test_discarded_step_speech_is_recorded_as_heard_and_turn_length_is_fed_back():
    c, llm = _controller([
        _route("wrong_transfer"),
        _out(say="사기는 아니죠?"),                  # 조회 전에 답하려 함 → 버림 (도구가 정해질 때까지 붙잡아 둬서 안 나감)
        _out(say="확인해 볼게요.", tool={"name": "lookup_transfers", "args": {}}),
        _out(say="어떤 송금인지 볼게요."),
    ])
    s = call(CallContext(customer={"name": "윤가영", "transfers": []}))
    reply = c.respond(s, TurnInput("돈 잘못 보냈어요"))
    spoken = [e.data["text"] for e in s.trace.of("spoken")]
    assert spoken == ["확인해 볼게요.", "어떤 송금인지 볼게요."]                  # 조회 전 답은 고객에게 안 나감 (docs/09 G6)
    assert s.trace.of("spoken")[0].data.get("by") == "code"                       # 첫 조회 전 한마디는 코드가 바로
    assert [e.data["text"] for e in s.trace.of("unheard_drop")] == ["사기는 아니죠?", "확인해 볼게요."]
    assert reply.text == "확인해 볼게요. 어떤 송금인지 볼게요."                   # 기록은 고객이 들은 그대로 (docs/09 G3)
    assert "[이번 턴에 고객이 이미 들은 말 — 8자]" in llm.calls[3][-1]["content"]


def test_earlier_turn_tool_results_reach_next_turn_prompt():
    orders = {"name": "한지민", "orders": [{"order_id": "A3002", "item": "에어팟 케이스", "status": "상품 준비"}]}
    c, llm = _controller([
        _route("general"),
        _out(say="확인해 볼게요.", tool={"name": "lookup_orders", "args": {}}),   # slots에 아무것도 안 적음
        _out(say="어느 물건인가요?"),
        _out(say="에어팟 케이스는 상품 준비 중이에요."),
    ])
    s = call(CallContext(customer=orders))
    c.respond(s, TurnInput("물건이 안 왔어요"))
    assert "A3002" not in llm.calls[1][0]["content"]            # 이번 턴 결과는 메시지로만 (중복 없음)
    c.respond(s, TurnInput("에어팟 케이스요"))
    system = llm.calls[-1][0]["content"]
    assert "[이전 턴 도구 결과" in system and "A3002" in system and "1턴" in system


def test_inbound_call_starts_with_fixed_greeting():
    from csagent.agent.contracts import CallContext
    s = call(CallContext(direction="inbound", service="토스"))
    assert s.history == [("agent", "안녕하세요, 토스 고객센터 AI 상담원입니다. 통화 내용은 녹음됩니다. 무엇을 도와드릴까요?")]
    assert [e.kind for e in s.trace.events if e.kind != "call_open"] == ["greeting"]
    # 아웃바운드는 받는 쪽이 먼저 말한다 — 소개는 워커 지시문이 첫 응답에서
    assert call(CallContext(direction="outbound", service="토스쇼핑")).history == []


def test_unlisted_phone_number_removed_from_speech_and_record():
    from csagent.agent.guardrails.contacts import ContactBook
    book = ContactBook({"faq:4705#0": [{"phone": "1644-2944", "kind": "전화", "org": "토스쇼핑 고객센터",
                                         "when": "2영업일 이상 송장 흐름이 없을 때", "hours": "",
                                         "service": "토스쇼핑", "verified": True}]})
    llm = ScriptedLLM([_route("general"),
                       _out(say="토스쇼핑 고객센터 1644-2944나 판매자 1588-0000로 문의해 주세요.")])
    loop = TurnLoop(FakeRetriever(), llm, TOOLS, gate=ExecutionGate(), contacts=book)
    c = CallController(loop, Router(llm, WORKERS), WORKERS)
    s = call(CallContext(customer=CARDS))
    events = list(c.respond_stream(s, TurnInput("택배가 안 와요")))
    spoken = "".join(e.data["text"] for e in events if e.type == "delta")
    # 검색된 문서의 연락처는 그대로, 목록에 없는 번호만 지운다 — 들은 말과 기록이 같아야 한다
    assert spoken == "토스쇼핑 고객센터 1644-2944나 판매자로 문의해 주세요."
    assert s.history[-1] == ("agent", spoken)
    assert s.trace.of("number_blocked")[0].data["numbers"] == ["1588-0000"]
    # 워커 입력의 [참고 문서]에 누구 번호·언제 쓰는지가 붙는다
    worker_prompt = llm.calls[-1][0]["content"]
    assert "[연락처] 토스쇼핑 고객센터 1644-2944 — 2영업일 이상 송장 흐름이 없을 때" in worker_prompt


def test_contact_owner_comes_from_verified_table_not_llm_label():
    from csagent.agent.guardrails.contacts import ContactBook
    # faq:4283 실측: 「고객센터(1661-7654)」를 LLM이 「토스 고객센터」로 붙임 → 표의 주인(토스뱅크)으로
    book = ContactBook({"x": [{"phone": "1661-7654", "kind": "전화", "org": "토스 고객센터", "when": "",
                               "hours": "", "service": "토스뱅크", "verified": True},
                              {"phone": "02-6991-0210", "kind": "팩스", "org": "토스 고객센터", "when": "",
                               "hours": "", "service": "토스", "verified": False}]})
    assert book.render("x") == "[연락처] 토스뱅크 고객센터 1661-7654"   # 팩스는 빠진다
    assert book.allowed(["x"]) == {"16617654"}


def test_card_loss_speaks_only_toss_family_numbers():
    from csagent.agent.guardrails.contacts import ContactBook, SpeechScreen, strip_unlisted
    # 10/16 회귀 실측: 카드 신청 FAQ(faq:5047)의 카드사 번호가 검색돼 「현대카드 고객센터 1577-6000」을 말함
    book = ContactBook({"faq:5047#0": [
        {"phone": "1577-6000", "kind": "전화", "org": "현대카드", "when": "카드 신청 상태", "hours": "",
         "service": "", "verified": False},
        {"phone": "1661-7654", "kind": "전화", "org": "토스뱅크 고객센터", "when": "", "hours": "",
         "service": "토스뱅크", "verified": True}]})
    assert WORKERS["card_loss"].policy.toss_numbers_only
    allowed = book.allowed(["faq:5047#0"], toss_only=True)
    assert allowed == {"16617654"}
    assert "1577-6000" not in book.render("faq:5047#0", toss_only=True)   # [참고 문서]의 [연락처] 줄에도 안 보인다
    assert book.allowed(["faq:5047#0"]) == {"15776000", "16617654"}      # 다른 업무는 그대로

    say = "현대카드 고객센터 1577-6000으로 연락해 주세요. 토스뱅크는 1661-7654로 연락 주세요."
    text, removed = strip_unlisted(say, allowed)
    # 번호를 지우면 조사도 맞춘다 (「고객센터으로」가 아니라 「고객센터로」)
    assert text == "현대카드 고객센터로 연락해 주세요. 토스뱅크는 1661-7654로 연락 주세요."
    assert removed == ["1577-6000"]
    # 스트리밍으로 나간 말도 기록과 같아야 한다 — 번호 앞 글자가 앞 조각에 있어도, 같은 조각에 있어도
    for pieces in (["현대카드 고객센터", " 1577-6000으로 연락해 주세요.", " 토스뱅크는 1661-7654로 연락 주세요."],
                   [say]):
        screen = SpeechScreen(allowed)
        assert "".join(screen.feed(p) for p in pieces) + screen.flush() == text


def test_removed_number_keeps_euro_after_final_consonant():
    from csagent.agent.guardrails.contacts import strip_unlisted
    # 「으로/로」는 번호 앞말의 받침으로 고른다 (받침 없음·ㄹ → 로, 그 밖 → 으로)
    assert strip_unlisted("카드사 대표 전화 1588-0000으로 거세요.", set())[0] == "카드사 대표 전화로 거세요."
    assert strip_unlisted("카드사 본점 1588-0000으로 거세요.", set())[0] == "카드사 본점으로 거세요."
    assert strip_unlisted("카드사 1588-0000으로 거세요.", set())[0] == "카드사로 거세요."
    assert strip_unlisted("카드사 앱 1588-0000으로.", set())[0] == "카드사 앱으로."
    assert strip_unlisted("카드사 채널 1588-0000으로.", set())[0] == "카드사 채널로."


def test_card_loss_masks_card_company_numbers_in_reference_docs():
    from csagent.agent.guardrails.contacts import mask_unlisted
    # 지시문으로 막아도 본문 번호를 읽어 말하려다 지워져 「번호는입니다」가 됨 → 본문에서 가린다
    body = "[카드사 고객센터 연락처] - 현대카드 1577-6000 - 토스뱅크 1661-7654"
    assert mask_unlisted(body, {"16617654"}) == "[카드사 고객센터 연락처] - 현대카드 (번호 생략) - 토스뱅크 1661-7654"


def test_greeting_not_in_worker_history():
    # 인사말이 assistant 메시지로 들어가면 모델이 막힐 때 통화 중간에 그대로 베낀다 (runs/20261016_183622_C)
    llm = ScriptedLLM([_route("card_loss"), _out(say="확인해 볼게요.", tool={"name": "lookup_cards", "args": {}}),
                       _out(say="조회됐어요.")])
    loop = TurnLoop(FakeRetriever(), llm, TOOLS, gate=ExecutionGate())
    c = CallController(loop, Router(llm, WORKERS), WORKERS)
    s = call(CallContext(direction="inbound", service="토스", customer=CARDS))
    list(c.respond_stream(s, TurnInput("카드 잃어버렸어요")))
    worker_msgs = llm.calls[-1]
    assert not any("무엇을 도와드릴까요" in m["content"] for m in worker_msgs if m["role"] == "assistant")
    assert worker_msgs[1]["role"] == "user"


def _out_confirm(say, confirm, answer=None, tool=None):
    return json.dumps({"intent": "continue", "pending_answer": answer, "slots": {}, "tool": tool,
                       "say": say, "confirm": confirm}, ensure_ascii=False)


def test_spoken_confirmation_is_registered_so_customer_is_asked_once():
    # card_loss choose: 말로 「분실신고할까요?」 → 동의 → 도구 → 게이트가 또 확인 = 확인 두 번 (runs/20261016_183622_C)
    report = {"name": "report_card_lost", "args": {"card_id": "C1"}}
    c, llm = _controller([
        _route("card_loss"),
        _out(say="확인해 볼게요.", tool={"name": "lookup_cards", "args": {}}),
        _out_confirm("토스유스카드 1234 분실신고할까요?", report),
        _out_confirm("분실신고 접수해 드릴게요.", None, answer={"answer": "yes", "quote": "네"}, tool=report),
        _out(say="분실신고가 접수됐어요.")])
    s = call(CallContext(customer=CARDS))
    s.authenticated = True
    c.respond(s, TurnInput("카드 잃어버렸어요. 못 찾을 것 같아요"))
    assert s.pending is not None and s.pending.has("report_card_lost", {"card_id": "C1"})
    assert s.trace.of("spoken_confirm")[0].data["registered"]
    c.respond(s, TurnInput("네 해주세요"))
    assert [e.data["name"] for e in s.trace.of("tool_result")][-1] == "report_card_lost"
    assert s.history[-1][1].endswith("분실신고가 접수됐어요.")


def test_spoken_confirmation_not_registered_before_auth():
    # 인증 전에는 동의를 받아도 실행할 수 없다 → 등록하지 않는다
    c, llm = _controller([
        _route("card_loss"),
        _out(say="확인해 볼게요.", tool={"name": "lookup_cards", "args": {}}),
        _out_confirm("분실신고할까요?", {"name": "report_card_lost", "args": {"card_id": "C1"}})])
    s = call(CallContext(customer=CARDS))
    c.respond(s, TurnInput("카드 잃어버렸어요"))
    assert s.pending is None
    assert s.trace.of("spoken_confirm")[0].data["registered"] is False


def test_end_call_blocked_in_turn_that_executed_then_allowed_next_turn():
    # h4 실측: 실행 결과 안내 + 「다른 도움 필요하시면」 + end_call을 한 턴에 내 고객이 다음 용건을 못 꺼냄
    c, _ = _controller([
        _route("card_loss"),
        _out(say="확인해 볼게요.", tool={"name": "lookup_cards", "args": {}}),
        _out(say="토스유스카드 1234 맞으시죠?"),
        _out(say="인증 요청을 보낼게요.", tool={"name": "request_app_auth", "args": {}}),
        _out(say="", tool={"name": "suspend_card", "args": {"card_id": "C1"}}),
        _out(say="토스유스카드 1234를 일시정지할까요?"),
        _out(say="", tool={"name": "suspend_card", "args": {"card_id": "C1"}},
             answer={"answer": "yes", "quote": "네 해 주세요"}),
        _out(intent="done", say="일시정지됐어요. 다른 도움 필요하시면 말씀해 주세요.",
             tool={"name": "end_call", "args": {"disposition": "resolved"}}),
        _route("general"),   # 4턴: 카드 업무는 3턴에 끝남 → 다음 말은 라우터가 다시 본다 (두 번째 용건이 들어오는 길)
        _out(intent="done", say="네, 감사합니다.", tool={"name": "end_call", "args": {"disposition": "resolved"}}),
    ])
    s = call(CallContext(customer=CARDS))
    c.respond(s, TurnInput("카드를 잃어버렸어요"))
    c.respond(s, TurnInput("네 일시정지해 주세요"))
    reply = c.respond(s, TurnInput("네 해 주세요"))
    assert s.succeeded and not reply.end_call
    assert reply.text.endswith("다른 도움 필요하시면 말씀해 주세요.")   # 막힌 스텝의 결과 안내는 버리지 않는다
    assert s.trace.of("gate")[-1].data["reason"] == "result_unheard"
    reply = c.respond(s, TurnInput("아니요 없어요"))
    assert reply.end_call


def test_question_with_tool_waits_for_answer():
    # o1_h6 실측: 「알려 드릴까요?」 + 조회 → 결과 받고 「네, 확인했습니다」로 스스로 답함
    c, llm = _controller([
        _route("card_loss"),
        _out(say="카드 상태를 확인해 드릴까요?", tool={"name": "lookup_cards", "args": {}}),
        _out(say="이 말은 나가면 안 됨"),
    ])
    s = call(CallContext(customer=CARDS))
    reply = c.respond(s, TurnInput("카드를 잃어버렸어요"))
    assert reply.text == "카드 상태를 확인해 드릴까요?"          # 조회는 하고 턴을 끝낸다
    assert [t.name for t in reply.tool_calls] == ["lookup_cards"]
    assert s.trace.of("question_hold")[-1].data["ran"] is True


def test_question_with_end_call_does_not_hang_up():
    c, _ = _controller([
        _route("card_loss"),
        _out(intent="done", say="다른 도움 필요하신가요?", tool={"name": "end_call", "args": {"disposition": "resolved"}}),
    ])
    s = call(CallContext(customer=CARDS))
    reply = c.respond(s, TurnInput("카드 분실 문의요"))
    assert not reply.end_call and reply.text == "다른 도움 필요하신가요?" and not reply.tool_calls


def test_question_with_execution_registers_confirmation_instead_of_running():
    c, _ = _controller([
        _route("card_loss"),
        _out(say="확인해 볼게요.", tool={"name": "lookup_cards", "args": {}}),
        _out(say="토스유스카드 1234 맞으시죠?"),
        _out(say="인증 요청을 보낼게요.", tool={"name": "request_app_auth", "args": {}}),
        _out(say="토스유스카드 1234를 일시정지할까요?", tool={"name": "suspend_card", "args": {"card_id": "C1"}}),
        _out(say="", tool={"name": "suspend_card", "args": {"card_id": "C1"}},
             answer={"answer": "yes", "quote": "네"}),
        _out(say="일시정지됐어요."),
    ])
    s = call(CallContext(customer=CARDS))
    c.respond(s, TurnInput("카드를 잃어버렸어요"))
    reply = c.respond(s, TurnInput("네 일시정지해 주세요"))
    assert reply.text.endswith("일시정지할까요?") and not s.succeeded
    assert s.trace.of("spoken_confirm")[-1].data["registered"] is True
    c.respond(s, TurnInput("네"))
    assert s.succeeded                                        # 동의 한 번으로 실행 (확인 두 번 없음)
