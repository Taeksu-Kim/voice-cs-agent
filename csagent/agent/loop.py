"""공통 대화 루프 — 한 턴 안에서: 검색 → LLM(스트리밍) → 확인 판정 → 도구(게이트 경유) → 반복.

기준선(A0/A1)과 C'의 워커가 이 루프 하나를 같이 쓴다. 차이는 TurnPolicy(업무 지시·허용 도구·슬롯)뿐이다.
턴 번호 증가·대화 이력 기록·업무 배정은 호출하는 쪽(에이전트·컨트롤러)이 한다.
"""

from __future__ import annotations

import difflib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from csagent.agent import prompts
from csagent.agent.contracts import AgentEvent, AgentReply, ToolCall
from csagent.agent.guardrails.contacts import ContactBook, SpeechScreen, mask_unlisted, strip_unlisted
from csagent.agent.guardrails.links import strip_links
from csagent.agent.guardrails.gate import ExecutionGate
from csagent.agent.state import CallState
from csagent.agent.streaming import SayExtractor, announced_done, unkept_promise
from csagent.agent.tools.registry import ToolRegistry, execution_key
from csagent.llm import ChatLLM, LLMResult, parse_json
from csagent.obs import event
from csagent.retrieval.filters import SearchFilter
from csagent.agent.protocol import Protocol, Step, step_text
from csagent.retrieval.retriever import HybridRetriever

log = logging.getLogger("csagent.agent")

STALL_SIMILARITY = 0.85
MAX_EARLIER_RESULTS, EARLIER_RESULT_CHARS = 8, 800   # 다음 턴 프롬프트에 넘기는 이전 도구 결과
TURN_SPEECH_CHARS = 120   # 고객이 대답하기 전까지 한 턴에 말하는 전체의 목표 길이 (prompts.py 말하기 규칙과 같은 값)
FALLBACK = "죄송해요, 잘 못 들었어요. 한 번만 다시 말씀해 주시겠어요?"
# 도구를 시도했는데 고객에게 한마디도 못 한 턴 (막힌 도구 되풀이 등) — 고객은 제대로 말했으니 「잘 못 들었어요」가 아니다
TOOL_STUCK = "죄송해요, 확인하는 중에 문제가 생겼어요. 한 번만 다시 말씀해 주시겠어요?"
LOOKUP_LINE = "확인해 볼게요."   # 첫 조회 전 코드가 하는 말 (docs/09 G6) — 지시문의 도구 스텝 한마디와 같다
PROMISE_NUDGE = ("[절차] 방금 확인해 보겠다고 말했지만 도구를 부르지 않았습니다. 이미 한 말은 고객이 들었습니다 — 다시 말하지 마세요. "
                 "확인할 도구가 있으면 지금 부르고 say는 비우세요. 쓸 수 있는 도구로 확인할 수 없는 일이면 확인할 수 없다고 짧게 말하세요.")
_DISPOSITIONS = {"send_app_link": "link_sent", "transfer_to_agent": "transferred"}

# 워커 출력 형식을 디코딩 단계에서 강제한다 (prompts의 [출력 형식]과 같은 키·순서).
# tool이 say 앞이다 — 말을 내보내기 전에 막힐 도구인지 안다 (docs/09 G7: 「진행할게요」 스트리밍 → 동의 확인에 걸림 →
# 「진행할까요?」가 실측 95턴). 대가는 도구 스텝의 첫 소리가 도구 JSON만큼 늦어지는 것 (도구 없는 스텝은 `null` 몇 토큰).
# json_object만 주면 대화 이력의 {"say": ...}를 흉내 내 intent·tool을 빼먹고 직전 말을 되풀이하는 일이 있었다.
WORKER_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"enum": ["continue", "switch", "add", "done"]},
        "pending_answer": {"anyOf": [
            {"type": "null"},
            {"type": "object",
             "properties": {"answer": {"enum": ["yes", "no", "unclear"]}, "quote": {"type": "string"}},
             "required": ["answer", "quote"]}]},
        "slots": {"type": "object"},
        "tool": {"anyOf": [
            {"type": "null"},
            {"type": "object",
             "properties": {"name": {"type": "string"}, "args": {"type": "object"}},
             "required": ["name", "args"]}]},
        "say": {"type": "string"},
        # 도구 없이 말로 실행 동의를 물을 때(「분실신고할까요?」) 그 실행 — 코드가 확인 질문으로 등록한다.
        # 말로 묻고 → 동의 → 도구 → 게이트가 또 확인 = 확인 두 번 (card_loss choose, 10/13·10/16 재발)을 구조로 막는다
        "confirm": {"anyOf": [
            {"type": "null"},
            {"type": "object",
             "properties": {"name": {"type": "string"}, "args": {"type": "object"}},
             "required": ["name", "args"]}]},
    },
    "required": ["intent", "pending_answer", "slots", "tool", "say", "confirm"],
}
# 말이 도구 앞인 순서 — 막힐 수 있는 도구가 없는 단계에서 쓴다 (docs/09 G7 혼합). 도구 순서를 늘 앞에 두면
# 도구 턴 첫 소리가 3.5 → 4.8초 (전체 회귀 202026). 조회·일반 발송 단계는 막힐 일이 없어 기다릴 이유가 없다
WORKER_SCHEMA_SAY_FIRST = {**WORKER_SCHEMA, "properties": {
    k: WORKER_SCHEMA["properties"][k] for k in ("intent", "pending_answer", "slots", "say", "tool", "confirm")}}


def _confirms_now(state: CallState, answer: Optional[dict], utterance: str, action: str, target: dict) -> bool:
    """이번 스텝의 pending_answer가 걸어 둔 확인 질문을 동의로 확정할 것인가 (`CallState.resolve_pending`과 같은 대조)."""
    p, answer = state.pending, answer or {}
    quote = str(answer.get("quote", "")).strip()
    return bool(p and p.kind == "confirm" and p.turn < state.turn and answer.get("answer") == "yes"
                and quote and quote in utterance and p.has(action, target))


def _tool_name(text: str) -> str:
    try:
        return (parse_json(text).get("tool") or {}).get("name", "")
    except (ValueError, json.JSONDecodeError, AttributeError):
        return ""


def _has_tool(text: str) -> bool:
    """워커 출력에 도구 호출이 있는가 — 붙잡아 둔 말을 내보낼지 정할 때 (형식이 깨졌으면 없는 것으로)."""
    try:
        return bool(parse_json(text).get("tool"))
    except (ValueError, json.JSONDecodeError, AttributeError):
        return False


@dataclass(frozen=True)
class SlotRule:
    """도구를 슬롯 값에 따라 허용한다 — 위험한 갈래는 LLM이 사실(슬롯)만 판단하고, 갈래 선택은 코드가 지킨다.
    예: 착오송금 접수 링크는 transfer_reason이 「실수」일 때만."""
    tool: str
    slot: str
    allowed: tuple[str, ...]
    message: str                               # 막혔을 때 LLM에게 줄 안내 ({value} = 현재 슬롯 값)


@dataclass(frozen=True)
class TurnPolicy:
    name: str
    task: str                                  # [이번 업무] 지시문
    tools: Optional[tuple[str, ...]] = None    # 허용 도구 (None = 전부)
    retrieve: bool = True
    first_tool: Optional[str] = None           # 이 업무에서 답하기 전에 반드시 먼저 할 조회 (코드가 강제)
    slot_rules: tuple[SlotRule, ...] = ()
    protocol: Optional[Protocol] = None        # 있으면 지시문·허용 도구를 지금 단계 것으로 쓴다 (task·tools 대신)
    toss_numbers_only: bool = False            # 말해도 되는 번호를 토스 계열 확인 번호로 좁힌다 (카드사 번호 등은 문서에 있어도 지움)


@dataclass
class LoopOutcome:
    reply: AgentReply
    intent: str = "continue"                   # continue | switch | add | done (마지막 스텝 기준)
    added: bool = False                        # 이 턴 어느 스텝에서든 add가 나왔나 (뒤 스텝이 continue로 덮어도 유지)
    slots: dict[str, Any] = field(default_factory=dict)


class TurnLoop:
    def __init__(self, retriever: HybridRetriever, llm: ChatLLM, tools: ToolRegistry, *,
                 gate: Optional[ExecutionGate] = None, k: int = 4, doc_chars: int = 700,
                 max_steps: int = 4, contacts: Optional[ContactBook] = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.tools = tools
        self.gate = gate
        self.k = k
        self.doc_chars = doc_chars
        self.max_steps = max_steps
        self.contacts = contacts      # 없으면 연락처 줄·번호 검사 없음

    def run(self, state: CallState, utterance: str, policy: TurnPolicy,
            slots: Optional[dict] = None, t0: Optional[float] = None) -> Iterator[AgentEvent | LoopOutcome]:
        """이벤트를 내보내고, 마지막에 LoopOutcome을 하나 내보낸다."""
        t0 = t0 or time.perf_counter()
        state.trace.where = policy.name if policy.protocol is None else \
            f"{policy.name}/{policy.protocol.current(state).id}"   # 좌표의 절차 축 (trace.py) — 검색 이벤트부터 맞게
        hits = self._retrieve(state, utterance, policy)
        t_retrieve = (time.perf_counter() - t0) * 1000
        if hits is not None:
            yield AgentEvent("retrieval", {
                "ms": round(t_retrieve), "worker": policy.name,
                "docs": [{"chunk_id": h.chunk_id, "title": h.title, "section": h.section,
                          "score": round(h.score, 4), "url": h.url} for h in hits]})

        # 말해도 되는 전화번호 = 이 통화에서 지금까지 검색된 문서들의 연락처
        allowed = None if self.contacts is None else self.contacts.allowed(
            (cid for e in state.trace.of("retrieval") for cid in e.data.get("chunk_ids", [])),
            toss_only=policy.toss_numbers_only)
        stage = self._enter_step(state, policy, None)
        messages = self._messages(state, utterance, policy, hits or [], slots or {})
        reply = AgentReply(text="", retrieved=[h.chunk_id for h in hits or []])
        outcome = LoopOutcome(reply)
        said: list[str] = []    # 스텝별 말 (같은 말 두 번 막기용)
        heard: list[str] = []   # 고객이 실제로 들은 말 — 기록·다음 턴 이력·평가는 이것 (docs/09 G3)
        nudged = False   # docs/09 G2 — 한 턴에 한 번만
        looked_up_said = False   # docs/09 G6 — 첫 조회 전 「확인해 볼게요」는 한 번만
        llm_ms, first_token_ms, tokens = 0.0, None, {"prompt": 0, "completion": 0}

        for step in range(1, self.max_steps + 1):
            yield AgentEvent("step", {"worker": policy.name, "step": step})   # 화면에 「LLM 생성 중」 표시용
            state.trace.step += 1   # 좌표의 스텝 — 한 턴에 워커가 바뀌어도 이어 센다 (trace.py)
            extractor, result, spoken, closed = SayExtractor(), None, "", False
            screen = SpeechScreen(allowed)
            # 첫 조회 전 — 조회는 코드가 강제하므로 이 스텝의 모델 말은 내보내지 않는다 (docs/09 G6).
            # 도구 없이 답하면 버릴 말이고(s4_b0_friend: 조회 안 한 조언 + 조회 뒤 같은 조언), 도구 스텝이면 「확인해 볼게요」 한마디다.
            # 대신 코드가 그 한마디를 바로 내보낸다 — 붙잡기만 하면 버리고 다시 생성하는 동안 첫 소리가 17~20초 걸렸다
            hold, unheard = self._needs_first_tool(state, policy), ""
            # 막힐 도구를 부르는 스텝의 말은 내보내지 않는다 (docs/09 G7) — tool이 say 앞이라 첫 조각 때 판정할 수 있다
            withhold, withheld = None, ""
            if hold and not looked_up_said:
                looked_up_said = True
                first_token_ms = first_token_ms or (time.perf_counter() - t0) * 1000
                state.trace.add(state.turn, "spoken", {"worker": policy.name, "text": LOOKUP_LINE, "by": "code"})
                heard.append(LOOKUP_LINE)
                yield AgentEvent("delta", {"text": LOOKUP_LINE})
                yield AgentEvent("say_end", {})
            tool_first = self._gated(state, policy)
            schema = WORKER_SCHEMA if tool_first else WORKER_SCHEMA_SAY_FIRST
            for item in self.llm.stream_chat(messages, max_tokens=320, json_mode=True, schema=schema):
                if isinstance(item, LLMResult):
                    result = item
                    continue
                piece = screen.feed(extractor.feed(item))
                if extractor.done:
                    piece += screen.flush()   # say가 닫혔다 — 붙잡아 둔 문장 끝까지 검사해 내보낸다
                if piece and extractor.intent != "switch":   # 전환이면 이 워커의 말은 고객에게 보내지 않는다
                    if hold:
                        unheard += piece
                        continue
                    if withhold is None:
                        withhold = self._would_block(state, policy, extractor.value("tool"),
                                                     {**(slots or {}), **outcome.slots},
                                                     extractor.value("pending_answer"), utterance)
                    if withhold:
                        withheld += piece
                        continue
                    first_token_ms = first_token_ms or (time.perf_counter() - t0) * 1000
                    spoken += piece
                    yield AgentEvent("delta", {"text": piece})
                if extractor.done and spoken and not closed:
                    # say가 닫혔다 — 음성은 뒤의 도구 인자 생성을 기다리지 않고 남은 조각을 바로 읽는다
                    closed = True
                    yield AgentEvent("say_end", {})
            if unheard:
                first = _tool_name(result.text) == policy.first_tool
                if first or not _has_tool(result.text):
                    # 첫 조회 스텝의 「확인해 볼게요」는 코드가 이미 했다 / 조회 없이 답한 말은 절차 강제로 버린다
                    state.trace.add(state.turn, "unheard_drop", {"worker": policy.name, "text": unheard,
                                                                 "tool": first})
                else:   # 다른 도구(연결·종료 등)를 부르는 스텝의 말은 고객이 들어야 한다 → 아래 release로
                    withheld, unheard = unheard, ""

            def release() -> Iterator[AgentEvent]:
                """붙잡아 둔 말이 결국 필요해졌다 (예측과 달리 실행됨·질문으로 남음) — 늦게라도 내보낸다."""
                nonlocal withheld, first_token_ms
                if not withheld:
                    return
                first_token_ms = first_token_ms or (time.perf_counter() - t0) * 1000
                state.trace.add(state.turn, "spoken", {"worker": policy.name, "text": withheld, "late": True})
                heard.append(withheld.strip())
                yield AgentEvent("delta", {"text": withheld})
                yield AgentEvent("say_end", {})
                withheld = ""
            if spoken:   # 스트리밍으로 이미 나간 말 — 뒤에서 버려도(절차 강제·도구 차단) 고객은 들었다
                state.trace.add(state.turn, "spoken", {"worker": policy.name, "text": spoken})
                heard.append(spoken.strip())
            llm_ms += result.latency_ms
            tokens["prompt"] += result.prompt_tokens
            tokens["completion"] += result.completion_tokens
            state.trace.add(state.turn, "llm", {"role": policy.name, "call_id": result.call_id, "tool_first": tool_first,
                                                 "ms": round(result.latency_ms),
                                                 "prompt_tokens": result.prompt_tokens,
                                                 "completion_tokens": result.completion_tokens})
            try:
                out = parse_json(result.text)
            except (ValueError, json.JSONDecodeError):
                event(log, "fallback", logging.WARNING, call_id=state.call_id, turn=state.turn,
                      worker=policy.name, reason="parse_error", llm_call_id=result.call_id,
                      output=result.text[:300])
                yield AgentEvent("delta", {"text": FALLBACK})
                said.append(FALLBACK)
                heard.append(FALLBACK)
                break

            outcome.intent = out.get("intent") or "continue"
            outcome.added |= outcome.intent == "add"
            if outcome.intent == "switch":
                state.trace.add(state.turn, "intent", {"worker": policy.name, "intent": "switch"})
                said.clear()
                break
            if isinstance(out.get("slots"), dict):
                outcome.slots.update(out["slots"])
            self._apply_pending_answer(state, out.get("pending_answer"), utterance)
            say, tool = (out.get("say") or "").strip(), out.get("tool")
            say, removed = strip_unlisted(say, allowed)   # 스트리밍에서 지운 것과 같은 결과 (기록되는 말)
            if removed:
                state.trace.add(state.turn, "number_blocked", {"worker": policy.name, "numbers": removed})
                event(log, "number_blocked", logging.WARNING, call_id=state.call_id, turn=state.turn,
                      worker=policy.name, numbers=removed, llm_call_id=result.call_id)
            say, links = strip_links(say)   # 전화로 URL을 읽어 주지 않는다 — 링크는 send_app_link로 (스트리밍과 같은 결과)
            if links:
                state.trace.add(state.turn, "link_blocked", {"worker": policy.name, "urls": links})
                event(log, "link_blocked", logging.WARNING, call_id=state.call_id, turn=state.turn,
                      worker=policy.name, urls=links, llm_call_id=result.call_id)
            if not tool and self._needs_first_tool(state, policy):
                # 절차 강제: 업무의 첫 조회를 하기 전에 답하려 하면 되돌려 보낸다 (이 응답은 버림)
                # 버린 응답은 메시지에 남기지 않는다 — 남기면 조회 뒤에도 그 응답의 질문을 되풀이한다.
                # 이 스텝의 말은 붙잡아 두었으므로(hold) 고객도 듣지 않았다
                state.trace.add(state.turn, "step_enforced", {"worker": policy.name, "tool": policy.first_tool})
                event(log, "step_enforced", call_id=state.call_id, turn=state.turn, worker=policy.name,
                      tool=policy.first_tool, llm_call_id=result.call_id)
                messages[-1] = {**messages[-1], "content": messages[-1]["content"]
                                + f"\n\n[절차] 답하기 전에 먼저 {policy.first_tool}로 사실을 확인하세요."}
                continue
            if say and say not in said:     # 같은 턴에 똑같은 문장을 두 번 말하지 않는다 (문자열 비교)
                said.append(say)
            if not tool:
                if say and isinstance(out.get("confirm"), dict):
                    self._spoken_confirm(state, policy, out["confirm"])
                if say and not nudged and step < self.max_steps and unkept_promise(say):
                    # 「확인해 볼게요」로 끝내면 고객은 결과를 기다리며 침묵한다 (docs/09 G2) → 한 번 되돌려 보낸다
                    nudged = True
                    state.trace.add(state.turn, "promise_nudge", {"worker": policy.name, "say": say[-60:]})
                    event(log, "promise_nudge", call_id=state.call_id, turn=state.turn, worker=policy.name,
                          llm_call_id=result.call_id)
                    messages += [{"role": "assistant", "content": result.text},
                                 {"role": "user", "content": PROMISE_NUDGE}]
                    continue
                break
            held = self._question_hold(state, policy, say, tool)
            if held:   # 질문으로 남는 말이다 — 고객이 들어야 한다
                yield from release()
            if held == "drop":
                break

            call, feedback, stop = self._run_tool(state, tool, policy, {**(slots or {}), **outcome.slots})
            self._log_tool(state, policy, call)
            spec = self.tools.for_call(state).get(call.name)
            if call.result.get("ok") and spec and spec.asks_customer and say \
                    and call.result.get("status") == "pending":
                stop = True  # 이 스텝의 말이 곧 질문이다 — 결과를 받고 또 말하면 같은 질문을 반복한다
            stop = stop or held == "stop"
            if call.result.get("ok") or call.result.get("blocked") == "result_unheard":
                yield from release()
            elif withheld:   # 예측대로 막혔다(동의 확인 등) — 고객은 「진행할게요」를 듣지 않았다
                state.trace.add(state.turn, "unheard_drop", {"worker": policy.name, "text": withheld,
                                                             "blocked": call.result.get("blocked")
                                                             or ("confirmation" if "confirmation" in call.result else "failed")})
                withheld = ""
            done_line = call.result.get("ok") and announced_done(call.name, " ".join(heard))
            if not call.result.get("ok") and said and said[-1] == say and not held \
                    and call.result.get("blocked") != "result_unheard":
                said.pop()   # 도구가 실패·차단된 스텝의 말은 그 도구가 된다는 전제였으므로 버린다 (재시도 스텝이 다시 말함)
            reply.tool_calls.append(call)
            yield AgentEvent("tool_call", {"name": call.name, "args": call.args})
            yield AgentEvent("tool_result", {"name": call.name, "result": call.result})
            self._apply_disposition(reply, call)
            if done_line:   # 미리 알린 발송·예약이 됐다 — 결과 스텝(LLM) 없이 완료 한마디로 턴을 끝낸다
                said.append(done_line)
                heard.append(done_line)
                yield AgentEvent("delta", {"text": done_line})
                break
            if stop:
                break
            if stage is not None and (new := self._enter_step(state, policy, stage)) is not stage:
                stage = new   # 도구 결과로 단계가 바뀌었다 → 새 단계의 지시·도구를 알려 준다
                feedback += "\n\n" + step_text(policy.protocol, stage)
            if heard:   # 한 턴에 여러 스텝이면 앞 스텝에서 한 말을 다시 하기 쉽다 → 고객이 이미 들은 말과 길이를 알려 준다
                # (실패·차단된 스텝의 말도 고객은 들었다 — 숨기면 「진행할게요. 진행할까요?」처럼 되풀이·모순된다, docs/09 G3)
                so_far = " ".join(heard)
                feedback += (f"\n[이번 턴에 고객이 이미 들은 말 — {len(so_far)}자] {so_far}\n"
                             "같은 내용을 다시 말하지 말고, 이 결과에 이어서 할 말만 say에 넣으세요. "
                             f"고객이 대답하기 전까지 말하는 전체가 {TURN_SPEECH_CHARS}자 안쪽이 되게 하세요.")
            messages += [{"role": "assistant", "content": result.text},
                         {"role": "user", "content": feedback}]
        else:
            event(log, "max_steps", logging.WARNING, call_id=state.call_id, turn=state.turn,
                  worker=policy.name, steps=self.max_steps)

        if outcome.intent != "switch" and not heard:
            # 고객이 들은 말이 하나도 없을 때만 — 말이 나간 뒤에 「잘 못 들었어요」를 붙이지 않는다
            line = TOOL_STUCK if reply.tool_calls else FALLBACK
            event(log, "fallback", logging.WARNING, call_id=state.call_id, turn=state.turn,
                  worker=policy.name, reason="tool_stuck" if reply.tool_calls else "no_say")
            heard.append(line)
            yield AgentEvent("delta", {"text": line})
        # 기록은 고객이 들은 그대로 — 버린 스텝의 말도 고객은 들었다 (평가·다음 턴 이력이 실제 통화와 같아야 한다)
        reply.text = " ".join(h for h in heard if h) if outcome.intent != "switch" else ""
        reply.timings_ms = {"retrieve": t_retrieve, "llm": llm_ms, "first_token": first_token_ms or 0.0,
                            "total": (time.perf_counter() - t0) * 1000}
        reply.tokens = tokens
        yield outcome

    def _gated(self, state: CallState, policy: TurnPolicy) -> bool:
        """이 단계에 실행 전에 막힐 수 있는 도구가 있나 — 있으면 도구를 먼저 생성해 막힐 말을 붙잡는다 (docs/09 G7).
        동의·인증·담당자 확인이 필요한 도구, 업무 규칙(slot_rule)이 걸린 도구, 결과를 모르는 실행이 남은 경우.
        종료·연결(handoff)은 모든 단계에 있어 빼고 본다 — 넣으면 모든 단계가 도구 먼저가 된다."""
        if state.unknown_executions:
            return True
        tools = self.tools.for_call(state)
        names = self._tools_for(state, policy)
        ruled = {r.tool for r in policy.slot_rules}
        for n in (names if names is not None else tools.names()):
            spec = tools.get(n)
            if spec is None or spec.risk == "handoff":
                continue
            if spec.requires_confirmation or spec.requires_auth or spec.requires_contact or n in ruled:
                return True
        return False

    def _would_block(self, state: CallState, policy: TurnPolicy, tool: Optional[dict], slots: dict,
                     answer: Optional[dict] = None, utterance: str = "") -> bool:
        """이 도구 호출이 실행 전에 막히는가 — `_run_tool`의 검사(허용 도구·근거·업무 규칙·게이트)를 부작용 없이 미리 본다.
        실행 중 실패(백엔드 장애)는 미리 알 수 없다. 이번 스텝의 pending_answer(동의)는 아직 반영 전이므로
        `resolve_pending`과 같은 대조로 「이번에 확정될 동의」를 미리 본다 — 안 보면 동의 직후 실행 스텝의 말이 매번
        실행 뒤로 밀렸다 (전체 회귀 11건, 도구 턴 첫 소리 +1.1초). 예측이 틀리면 release·unheard_drop이 바로잡는다."""
        if not tool:
            return False
        name, args = tool.get("name", ""), tool.get("args") or {}
        spec = self.tools.for_call(state).get(name)
        if spec is None:
            return True
        names = self._tools_for(state, policy)
        allowed = set(names) if names is not None else None
        in_policy = self.gate is None or allowed is None or name in allowed
        if in_policy and spec.precheck and spec.precheck(state, args):
            return True
        if any(r.tool == name and slots.get(r.slot) not in r.allowed for r in policy.slot_rules):
            return True
        if self.gate is not None:
            decision = self.gate.check(state, spec, args, allowed_tools=allowed)
            if decision.reason == "confirmation_required" and _confirms_now(state, answer, utterance, name, spec.target(args)):
                return False
            return not decision.allowed and decision.reason != "result_unheard"
        return False

    @staticmethod
    def _needs_first_tool(state: CallState, policy: TurnPolicy) -> bool:
        if not policy.first_tool or state.active is None:
            return False
        return policy.first_tool not in state.active.tools_done

    @staticmethod
    def _stalled(state: CallState) -> bool:
        """직전 두 에이전트 응답이 거의 같은 문장인가 (문자열 유사도 — 의미 판단 아님)."""
        said = [t for r, t in state.history if r == "agent"][-2:]
        return len(said) == 2 and difflib.SequenceMatcher(None, *said).ratio() >= STALL_SIMILARITY

    # -- 검색 --

    def _retrieve(self, state: CallState, utterance: str, policy: TurnPolicy):
        if not policy.retrieve:
            return None
        prev = [text for role, text in state.history if role == "customer"][-1:]
        hits = self.retriever.search(
            " ".join(prev + [utterance]), self.k,
            flt=SearchFilter(audience=state.ctx.audience, companies=state.ctx.companies))
        state.trace.add(state.turn, "retrieval", {  # 본문은 평가(근거 채점)용
            "worker": policy.name, "chunk_ids": [h.chunk_id for h in hits],
            "texts": [h.display_text()[:self.doc_chars] for h in hits]})
        return hits

    # -- 도구 --

    def _run_tool(self, state: CallState, tool: dict, policy: TurnPolicy,
                  slots: dict) -> tuple[ToolCall, str, bool]:
        """(호출 기록, LLM에게 돌려줄 메시지, 이번 턴을 끝낼지)."""
        name, args = tool.get("name", ""), tool.get("args") or {}
        call = ToolCall(name, args)
        state.trace.add(state.turn, "tool_request", {"name": name, "args": args, "worker": policy.name})
        tools = self.tools.for_call(state)
        spec = tools.get(name)
        if spec is None:
            call.result = {"ok": False, "error": f"unknown_tool:{name}"}
            return call, f"[도구 오류] 없는 도구입니다: {name}", False

        names = self._tools_for(state, policy)
        allowed = set(names) if names is not None else None
        # 이 워커가 쓸 수 없는 도구면 근거 검증을 건너뛰고 게이트가 not_allowed로 막는다 —
        # 「먼저 조회하라」는 근거 검증 피드백이 이 워커에 없는 도구 쪽으로 모델을 이끌지 않게
        in_policy = self.gate is None or allowed is None or name in allowed
        if in_policy and spec.precheck and (err := spec.precheck(state, args)):
            # 대상 값이 틀렸으면 게이트·확인 질문 전에 돌려보낸다 (틀린 대상으로 고객 동의를 받지 않게)
            call.result = {"ok": False, "blocked": "precheck", "error": err}
            state.trace.add(state.turn, "precheck", {"name": name, "error": err, "kind": "ungrounded_args"})
            return call, f"[인자 오류] {err}. 아직 아무것도 실행하지 않았습니다. 올바른 값으로 다시 호출하세요.", False

        for rule in policy.slot_rules:
            value = slots.get(rule.slot)
            if rule.tool == name and value not in rule.allowed:
                call.result = {"ok": False, "blocked": "slot_rule", "slot": rule.slot, "value": value}
                state.trace.add(state.turn, "slot_rule", {"name": name, "slot": rule.slot, "value": value})
                return call, f"[실행 차단] {rule.message.format(value=value)}", False

        if self.gate is not None:
            decision = self.gate.check(state, spec, args, allowed_tools=allowed)
            state.trace.add(state.turn, "gate", {"name": name, "allowed": decision.allowed,
                                                 "reason": decision.reason})
            if decision.reason == "tasks_queued" and state.end_blocked_turn is None:
                state.end_blocked_turn = state.turn   # 종료 차단은 이 턴 한 번만 — 다음 턴에 또 끝내려 하면 허용
            if not decision.allowed and decision.reason == "confirmation_required":
                return call, self._ask_confirmation(state, call, spec.name, spec.target(args)), False
            if decision.reason == "result_unheard":
                # 워커를 다시 부르지 않고 턴을 끝낸다 — 이 스텝의 말(결과 안내)은 그대로 두고 고객 대답을 기다린다
                call.result = {"ok": False, "blocked": decision.reason}
                return call, "", True
            if not decision.allowed:
                call.result = {"ok": False, "blocked": decision.reason}
                return call, f"[실행 차단] {decision.message}", False

        outcome = tools.run(state, name, args)
        call.result = {"ok": outcome.ok, **outcome.result,
                       **({"error": outcome.error} if outcome.error else {})}
        state.trace.add(state.turn, "tool_result", {"name": name, "risk": spec.risk, **call.result})
        if outcome.ok and spec.risk in ("execute", "notify"):   # 정지·분실신고, 링크 발송·재통화 예약 등
            state.acted_turn = state.turn
        if outcome.ok and spec.risk == "read":   # 지금 상태를 다시 봤다 — 결과를 몰랐던 실행도 이제 판단할 수 있다
            state.unknown_executions.clear()
        if not outcome.ok and spec.risk == "execute" and outcome.error.startswith("TimeoutError"):
            # 응답 없음 = 처리됐는지 모름. 「처리되지 않았습니다」라고 단정하던 것 (s5_h8, 10/17)과
            # 재시도로 두 번 실행될 위험을 막는다 (docs/09 G9)
            state.unknown_executions.add(execution_key(spec, args))
            call.result["outcome"] = "unknown"
            state.trace.add(state.turn, "execution_unknown", {"name": name, "args": args})
            # 다음 행동은 지금 단계에서 쓸 수 있는 도구로만 안내한다 — 조회가 막힌 단계(card_loss 동의 뒤)에서
            # 「조회로 확인하라」고 하면 그 조회가 다시 막혔다
            reads = [n for n in (names or tools.names())
                     if (s := tools.get(n)) is not None and s.risk == "read"]
            how = (f"{', '.join(reads)}로 지금 상태를 확인하거나 상담원에게 연결하세요" if reads
                   else "상담원에게 연결하세요 (transfer_to_agent)")
            return call, (f"[결과 확인 불가] {name} 요청에 백엔드 응답이 없어 처리됐는지 알 수 없습니다. "
                          "처리됐다고도, 안 됐다고도 말하지 마세요. 처리 여부를 지금 확인할 수 없다고 사실대로 알리고, "
                          f"{how}. 같은 실행을 바로 다시 하지 마세요."), False
        if outcome.ok and state.active is not None:
            state.active.tools_done.add(name)
            lists = [v for v in outcome.result.values() if isinstance(v, list)]
            if lists and not any(lists):
                state.active.tools_empty.add(name)
            else:
                state.active.tools_empty.discard(name)
        prefix = "[도구 결과]" if outcome.ok else "[도구 실패]"
        # 이관·종료는 성공했을 때만 턴을 끝낸다 (실패하면 오류를 보고 다시 시도하게)
        return call, f"{prefix} " + json.dumps(call.result, ensure_ascii=False), \
            spec.risk == "handoff" and outcome.ok

    @staticmethod
    def _log_tool(state: CallState, policy: TurnPolicy, call: ToolCall) -> None:
        """도구 호출 한 건의 결말: ok | confirm(확인 질문 걸림) | blocked(사유) | failed."""
        r = call.result or {}
        if r.get("ok"):
            status, level = "ok", logging.INFO
        elif "confirmation" in r:
            status, level = "confirm", logging.INFO
        elif r.get("blocked"):
            status, level = "blocked", logging.INFO
        else:
            status, level = "failed", logging.WARNING
        event(log, "tool", level, call_id=state.call_id, turn=state.turn, worker=policy.name,
              tool=call.name, status=status, reason=r.get("blocked") or r.get("error"))

    def _question_hold(self, state: CallState, policy: TurnPolicy, say: str, tool: dict) -> str:
        """말이 질문(「?」로 끝남)인데 같은 응답에 도구를 냈다 → 고객 대답을 기다린다.
        묻고 스스로 답하기(o1_h6 「알려 드릴까요? 네, 확인했습니다」)·묻고 끊기·묻고 대답 전에 예약하기가
        지시문으로 금지돼 있어도 샘 (10/15~17 기록 271통화 중 19건, 전부 「?」로 끝남) → 코드로 막는다.
        "" 그대로 진행 | "stop" 조회는 하고 턴 종료 (결과는 다음 턴에 씀) | "drop" 실행하지 않고 턴 종료."""
        name, args = tool.get("name", ""), tool.get("args") or {}
        spec = self.tools.for_call(state).get(name)
        if not say.endswith("?") or spec is None or spec.asks_customer:
            return ""
        state.trace.add(state.turn, "question_hold", {"name": name, "args": args,
                                                       "ran": spec.risk == "read"})
        if spec.risk == "read":
            return "stop"
        if spec.requires_confirmation:   # 「분실신고할까요?」 + 실행 → 그 질문을 확인 질문으로 등록 (동의하면 바로 실행)
            self._spoken_confirm(state, policy, {"name": name, "args": args})
        return "drop"

    def _spoken_confirm(self, state: CallState, policy: TurnPolicy, ask: dict) -> None:
        """말로 물은 실행 동의를 확인 질문으로 등록한다 — 게이트가 「동의만 없어서」 막을 실행일 때만.
        인증 전·대상 오류·허용 안 된 도구면 등록하지 않는다 (동의를 받아도 실행할 수 없는 질문이다)."""
        name, args = ask.get("name", ""), ask.get("args") or {}
        spec = self.tools.for_call(state).get(name)
        reason = ""
        if spec is None or not spec.requires_confirmation:
            reason = "no_confirmation_needed"
        elif spec.precheck and spec.precheck(state, args):
            reason = "precheck"
        elif self.gate is not None:
            tools = self._tools_for(state, policy)
            decision = self.gate.check(state, spec, args, allowed_tools=set(tools) if tools is not None else None)
            if decision.reason != "confirmation_required":
                reason = decision.reason or "already_confirmed"
        if reason:
            state.trace.add(state.turn, "spoken_confirm", {"name": name, "args": args, "registered": False,
                                                           "reason": reason})
            return
        state.ask_confirm(name, spec.target(args))
        state.trace.add(state.turn, "spoken_confirm", {"name": name, "args": args, "registered": True})

    @staticmethod
    def _ask_confirmation(state: CallState, call: ToolCall, action: str, target: dict) -> str:
        """동의 없는 실행 요청 → 실행하지 않고 확인 질문을 건다 (질문 걸기는 코드가 한다)."""
        status = state.ask_confirm(action, target)
        call.result = {"ok": False, "confirmation": status, "action": action, "target": target}
        if status == "waiting":
            return ("[확인 대기] 이 실행에 대한 확인 질문을 이미 걸어 두었고 아직 동의가 확정되지 않았습니다. "
                    "다시 묻지 말고, 이번 고객 말을 pending_answer(answer, quote)로 판정해서 같은 도구와 함께 출력하세요.")
        return (f"[확인 필요] 아직 실행하지 않았습니다. 고객 동의가 필요해 확인 질문을 걸었습니다: {action} "
                f"{json.dumps(target, ensure_ascii=False)}. 대상이 더 있으면 같은 방식으로 이어서 요청하고, "
                "다 걸었으면 tool 없이 무엇을 할지 말하고 진행할지 물어보세요.")

    @staticmethod
    def _apply_pending_answer(state: CallState, ans: Optional[dict], utterance: str) -> None:
        if state.pending is None or not isinstance(ans, dict):
            return
        state.resolve_pending(str(ans.get("answer", "unclear")), str(ans.get("quote", "")), utterance)

    @staticmethod
    def _apply_disposition(reply: AgentReply, call: ToolCall) -> None:
        if not call.result or not call.result.get("ok"):
            return
        if call.name == "end_call":
            reply.end_call = True
            reply.disposition = call.args.get("disposition", "resolved")
        elif call.name == "transfer_to_agent":
            reply.end_call = True
            reply.disposition = "transferred"
        elif call.name in _DISPOSITIONS:
            reply.disposition = _DISPOSITIONS[call.name]

    # -- 프로토콜 단계 --

    @staticmethod
    def _tools_for(state: CallState, policy: TurnPolicy) -> Optional[tuple[str, ...]]:
        if policy.protocol is not None:
            return policy.protocol.tools(policy.protocol.current(state))
        return policy.tools

    @staticmethod
    def _enter_step(state: CallState, policy: TurnPolicy, prev: Optional[Step]) -> Optional[Step]:
        """지금 단계를 정하고, 바뀌었으면 trace에 남긴다. 프로토콜이 없으면 None."""
        if policy.protocol is None:
            return None
        step = policy.protocol.current(state)
        state.trace.where = f"{policy.name}/{step.id}"
        if prev is not None and step.id == prev.id:
            return prev
        state.trace.add(state.turn, "protocol_step", {"worker": policy.name, "step": step.id})
        return step

    # -- 컨텍스트 --

    @staticmethod
    def _earlier_tool_results(state: CallState) -> str:
        """이전 턴들의 도구 결과. 이번 턴 결과는 메시지([도구 결과])로 이미 들어간다.

        대화 이력에는 상담원의 say만 남으므로, 이걸 넣지 않으면 다음 턴의 모델은 조회 결과를 볼 수 없다
        (모델이 slots에 자발적으로 옮겨 적었을 때만 보였다 → 안 적은 실행에서 없는 주문을 지어냄, s1_b2).
        같은 결과(반복 조회)는 한 번만, 최근 것부터 MAX_EARLIER_RESULTS개.
        """
        seen, lines = set(), []
        for e in reversed(state.trace.of("tool_result")):
            if e.turn >= state.turn or e.data.get("name") == "end_call":
                continue
            body = json.dumps({k: v for k, v in e.data.items() if k != "risk"}, ensure_ascii=False)
            if body in seen:
                continue
            seen.add(body)
            lines.append(f"- {e.turn}턴 {body[:EARLIER_RESULT_CHARS]}")
            if len(lines) >= MAX_EARLIER_RESULTS:
                break
        return "\n".join(reversed(lines)) or "(없음)"

    def _doc_text(self, hit, policy: TurnPolicy) -> str:
        text = hit.display_text()[:self.doc_chars]
        if policy.toss_numbers_only and self.contacts:   # 토스 계열 확인 번호만 남기고 본문의 다른 번호는 가린다
            text = mask_unlisted(text, self.contacts.allowed([hit.chunk_id], toss_only=True))
        return text

    def _contact_lines(self, chunk_id: str, toss_only: bool = False) -> str:
        lines = self.contacts.render(chunk_id, toss_only=toss_only) if self.contacts else ""
        return "\n" + lines if lines else ""

    def _messages(self, state: CallState, utterance: str, policy: TurnPolicy, hits,
                  slots: dict) -> list[dict]:
        docs = "\n\n".join(f"[{i}] {self._doc_text(h, policy)}" + self._contact_lines(h.chunk_id, policy.toss_numbers_only)
                           for i, h in enumerate(hits, 1)) or "(없음)"
        p = state.pending
        if p is None:
            pending = "(없음)"
        elif p.kind == "confirm":
            pending = "실행 동의: " + ", ".join(f"{a} {json.dumps(t, ensure_ascii=False)}" for a, t in p.items)
        else:
            pending = "담당자 본인인지"
        tools = self._tools_for(state, policy)
        names = list(tools) if tools is not None else None
        p_ = policy.protocol
        system = prompts.SYSTEM.format(
            service=state.ctx.service,
            task=p_.task_text(p_.current(state)) if p_ is not None else policy.task, tools=self.tools.for_call(state).prompt(names), pending=pending,
            queued=", ".join(t.name for t in state.queue) or "(없음)",
            slots=json.dumps(slots, ensure_ascii=False) if slots else "(없음)",
            tool_results=self._earlier_tool_results(state))
        context = prompts.CONTEXT.format(
            today=state.ctx.today or "-", audience=state.ctx.audience, direction=state.ctx.direction,
            name=state.ctx.customer.get("name", "-"), docs=docs,
            auth="완료" if state.authenticated else "안 됨",
            contact="완료" if state.contact_verified else "안 됨")
        if self._stalled(state):
            context += ("\n\n[진행 정체] 직전 두 응답이 거의 같은 말이었다. 같은 말을 되풀이하지 않는다. "
                        "더 진행할 수 없으면 이유를 한 문장으로 말하고 상담원 연결(transfer_to_agent)을 제안한다.")
        # Qwen 채팅 템플릿은 system 메시지를 맨 앞 하나만 허용한다 → 규칙과 통화 정보를 합친다
        msgs = [{"role": "system", "content": system + "\n\n" + context}]
        # 0턴 고정 인사는 빼다 — 모델이 자기 말로 보고 막힐 때 그대로 베꼈다 (s5_b7_auth_fail, runs/20261016_183622_C).
        # 인사를 이미 했다는 사실은 SYSTEM 규칙에 있다
        history = state.history[1:] if state.trace.of("greeting") else state.history
        for role, text in history:
            msgs.append({"role": "user" if role == "customer" else "assistant",
                         "content": text if role == "customer"
                         else json.dumps({"say": text}, ensure_ascii=False)})
        msgs.append({"role": "user", "content": utterance})
        return msgs

