"""공통 대화 루프 — 한 턴 안에서: 검색 → LLM(스트리밍) → 확인 판정 → 도구(게이트 경유) → 반복.

기준선(A0/A1)과 C'의 워커가 이 루프 하나를 같이 쓴다. 차이는 TurnPolicy(업무 지시·허용 도구·슬롯)뿐이다.
턴 번호 증가·대화 이력 기록·업무 배정은 호출하는 쪽(에이전트·컨트롤러)이 한다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from csagent.agent import prompts
from csagent.agent.contracts import AgentEvent, AgentReply, ToolCall
from csagent.agent.guardrails.gate import ExecutionGate
from csagent.agent.state import CallState
from csagent.agent.streaming import SayExtractor
from csagent.agent.tools.registry import ToolRegistry
from csagent.llm import ChatLLM, LLMResult, parse_json
from csagent.retrieval.filters import SearchFilter
from csagent.retrieval.retriever import HybridRetriever

FALLBACK = "죄송해요, 잘 못 들었어요. 한 번만 다시 말씀해 주시겠어요?"
_DISPOSITIONS = {"send_app_link": "link_sent", "transfer_to_agent": "transferred"}


@dataclass(frozen=True)
class TurnPolicy:
    name: str
    task: str                                  # [이번 업무] 지시문
    tools: Optional[tuple[str, ...]] = None    # 허용 도구 (None = 전부)
    retrieve: bool = True
    first_tool: Optional[str] = None           # 이 업무에서 답하기 전에 반드시 먼저 할 조회 (코드가 강제)


@dataclass
class LoopOutcome:
    reply: AgentReply
    intent: str = "continue"                   # continue | switch | done
    slots: dict[str, Any] = field(default_factory=dict)


class TurnLoop:
    def __init__(self, retriever: HybridRetriever, llm: ChatLLM, tools: ToolRegistry, *,
                 gate: Optional[ExecutionGate] = None, k: int = 4, doc_chars: int = 700,
                 max_steps: int = 4) -> None:
        self.retriever = retriever
        self.llm = llm
        self.tools = tools
        self.gate = gate
        self.k = k
        self.doc_chars = doc_chars
        self.max_steps = max_steps

    def run(self, state: CallState, utterance: str, policy: TurnPolicy,
            slots: Optional[dict] = None, t0: Optional[float] = None) -> Iterator[AgentEvent | LoopOutcome]:
        """이벤트를 내보내고, 마지막에 LoopOutcome을 하나 내보낸다."""
        t0 = t0 or time.perf_counter()
        hits = self._retrieve(state, utterance, policy)
        t_retrieve = (time.perf_counter() - t0) * 1000
        if hits is not None:
            yield AgentEvent("retrieval", {
                "ms": round(t_retrieve), "worker": policy.name,
                "docs": [{"chunk_id": h.chunk_id, "title": h.title, "section": h.section,
                          "score": round(h.score, 4)} for h in hits]})

        messages = self._messages(state, utterance, policy, hits or [], slots or {})
        reply = AgentReply(text="", retrieved=[h.chunk_id for h in hits or []])
        outcome = LoopOutcome(reply)
        said: list[str] = []
        llm_ms, first_token_ms, tokens = 0.0, None, {"prompt": 0, "completion": 0}

        for step in range(1, self.max_steps + 1):
            yield AgentEvent("step", {"worker": policy.name, "step": step})   # 화면에 「LLM 생성 중」 표시용
            extractor, result = SayExtractor(), None
            for item in self.llm.stream_chat(messages, max_tokens=320, json_mode=True):
                if isinstance(item, LLMResult):
                    result = item
                    continue
                piece = extractor.feed(item)
                if piece and extractor.intent != "switch":   # 전환이면 이 워커의 말은 고객에게 보내지 않는다
                    first_token_ms = first_token_ms or (time.perf_counter() - t0) * 1000
                    yield AgentEvent("delta", {"text": piece})
            llm_ms += result.latency_ms
            tokens["prompt"] += result.prompt_tokens
            tokens["completion"] += result.completion_tokens
            state.trace.add(state.turn, "llm", {"role": policy.name, "ms": round(result.latency_ms),
                                                 "prompt_tokens": result.prompt_tokens,
                                                 "completion_tokens": result.completion_tokens})
            try:
                out = parse_json(result.text)
            except (ValueError, json.JSONDecodeError):
                yield AgentEvent("delta", {"text": FALLBACK})
                said.append(FALLBACK)
                break

            outcome.intent = out.get("intent") or "continue"
            if outcome.intent == "switch":
                state.trace.add(state.turn, "intent", {"worker": policy.name, "intent": "switch"})
                said.clear()
                break
            if isinstance(out.get("slots"), dict):
                outcome.slots.update(out["slots"])
            self._apply_pending_answer(state, out.get("pending_answer"), utterance)
            say, tool = (out.get("say") or "").strip(), out.get("tool")
            if not tool and self._needs_first_tool(state, policy):
                # 절차 강제: 업무의 첫 조회를 하기 전에 답하려 하면 되돌려 보낸다 (이 응답은 버림)
                state.trace.add(state.turn, "step_enforced", {"worker": policy.name, "tool": policy.first_tool})
                messages += [{"role": "assistant", "content": result.text},
                             {"role": "user", "content": f"[절차] 답하기 전에 먼저 {policy.first_tool}로 사실을 확인하세요."}]
                continue
            if say and say not in said:     # 같은 턴에 똑같은 문장을 두 번 말하지 않는다 (문자열 비교)
                said.append(say)
            if not tool:
                break

            call, feedback, stop = self._run_tool(state, tool, policy)
            spec = self.tools.get(call.name)
            if call.result.get("ok") and spec and spec.asks_customer and say \
                    and call.result.get("status") == "pending":
                stop = True  # 이 스텝의 말이 곧 질문이다 — 결과를 받고 또 말하면 같은 질문을 반복한다
            if not call.result.get("ok") and said and said[-1] == say:
                said.pop()   # 도구가 실패·차단된 스텝의 말은 그 도구가 된다는 전제였으므로 버린다 (재시도 스텝이 다시 말함)
            reply.tool_calls.append(call)
            yield AgentEvent("tool_call", {"name": call.name, "args": call.args})
            yield AgentEvent("tool_result", {"name": call.name, "result": call.result})
            self._apply_disposition(reply, call)
            if stop:
                break
            if said:   # 한 턴에 여러 스텝이면 앞 스텝에서 한 말을 다시 하기 쉽다 → 이미 한 말을 알려 준다
                feedback += (f"\n[이번 턴에 이미 한 말] {' '.join(said)}\n"
                             "같은 내용을 다시 말하지 말고, 이 결과에 이어서 할 말만 say에 넣으세요.")
            messages += [{"role": "assistant", "content": result.text},
                         {"role": "user", "content": feedback}]

        if outcome.intent != "switch" and not said:
            said.append(FALLBACK)
            yield AgentEvent("delta", {"text": FALLBACK})
        reply.text = " ".join(said)
        reply.timings_ms = {"retrieve": t_retrieve, "llm": llm_ms, "first_token": first_token_ms or 0.0,
                            "total": (time.perf_counter() - t0) * 1000}
        reply.tokens = tokens
        yield outcome

    @staticmethod
    def _needs_first_tool(state: CallState, policy: TurnPolicy) -> bool:
        if not policy.first_tool or state.active is None:
            return False
        return policy.first_tool not in state.active.tools_done

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

    def _run_tool(self, state: CallState, tool: dict, policy: TurnPolicy) -> tuple[ToolCall, str, bool]:
        """(호출 기록, LLM에게 돌려줄 메시지, 이번 턴을 끝낼지)."""
        name, args = tool.get("name", ""), tool.get("args") or {}
        call = ToolCall(name, args)
        state.trace.add(state.turn, "tool_request", {"name": name, "args": args, "worker": policy.name})
        spec = self.tools.get(name)
        if spec is None:
            call.result = {"ok": False, "error": f"unknown_tool:{name}"}
            return call, f"[도구 오류] 없는 도구입니다: {name}", False

        if self.gate is not None:
            allowed = set(policy.tools) if policy.tools is not None else None
            decision = self.gate.check(state, spec, args, allowed_tools=allowed)
            state.trace.add(state.turn, "gate", {"name": name, "allowed": decision.allowed,
                                                 "reason": decision.reason})
            if not decision.allowed:
                call.result = {"ok": False, "blocked": decision.reason}
                return call, f"[실행 차단] {decision.message}", False

        outcome = self.tools.run(state, name, args)
        call.result = {"ok": outcome.ok, **outcome.result,
                       **({"error": outcome.error} if outcome.error else {})}
        state.trace.add(state.turn, "tool_result", {"name": name, "risk": spec.risk, **call.result})
        if outcome.ok and state.active is not None:
            state.active.tools_done.add(name)
        prefix = "[도구 결과]" if outcome.ok else "[도구 실패]"
        # 이관·종료는 성공했을 때만 턴을 끝낸다 (실패하면 오류를 보고 다시 시도하게)
        return call, f"{prefix} " + json.dumps(call.result, ensure_ascii=False), \
            spec.risk == "handoff" and outcome.ok

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

    # -- 컨텍스트 --

    def _messages(self, state: CallState, utterance: str, policy: TurnPolicy, hits,
                  slots: dict) -> list[dict]:
        docs = "\n\n".join(f"[{i}] {h.display_text()[:self.doc_chars]}"
                           for i, h in enumerate(hits, 1)) or "(없음)"
        p = state.pending
        pending = (f"{'실행 재확인' if p.kind == 'confirm' else '담당자 확인'}: "
                   f"{p.action} {json.dumps(p.target, ensure_ascii=False)}" if p else "(없음)")
        names = list(policy.tools) if policy.tools is not None else None
        system = prompts.SYSTEM.format(
            task=policy.task, tools=self.tools.prompt(names), pending=pending,
            slots=json.dumps(slots, ensure_ascii=False) if slots else "(없음)")
        context = prompts.CONTEXT.format(
            today=state.ctx.today or "-", audience=state.ctx.audience, direction=state.ctx.direction,
            name=state.ctx.customer.get("name", "-"), docs=docs,
            auth="완료" if state.authenticated else "안 됨",
            contact="완료" if state.contact_verified else "안 됨")
        # Qwen 채팅 템플릿은 system 메시지를 맨 앞 하나만 허용한다 → 규칙과 통화 정보를 합친다
        msgs = [{"role": "system", "content": system + "\n\n" + context}]
        for role, text in state.history:
            msgs.append({"role": "user" if role == "customer" else "assistant",
                         "content": text if role == "customer"
                         else json.dumps({"say": text}, ensure_ascii=False)})
        msgs.append({"role": "user", "content": utterance})
        return msgs
