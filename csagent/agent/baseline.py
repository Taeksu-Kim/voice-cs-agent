"""기준선 에이전트 (A0 / A1) — 단일 LLM이 모든 도구를 들고 매 스텝 「도구 / 답변」을 정한다.

- A0: gate=None — 실행 조건을 프롬프트로만 지킨다
- A1: gate=ExecutionGate() — 같은 프롬프트 + 실행 직전 코드 검사 (C'와 같은 게이트)

본체는 respond_stream(): 단계마다 이벤트를 내보낸다 (웹 서버 SSE가 그대로 전달).
respond()는 그 스트림을 끝까지 소비해 최종 AgentReply만 돌려준다 (평가 파이프라인용).
통화 상태(CallState)는 밖에서 들고 오고, 에이전트가 이번 턴의 변화(턴 번호·이력·확인·실행)를 기록한다.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Iterator, Optional

from csagent.agent import prompts
from csagent.agent.contracts import AgentEvent, AgentReply, ToolCall, TurnInput
from csagent.agent.guardrails.gate import ExecutionGate
from csagent.agent.state import CallState
from csagent.agent.streaming import SayExtractor
from csagent.agent.tools.registry import ToolRegistry
from csagent.llm import ChatLLM, LLMResult, parse_json
from csagent.retrieval.filters import SearchFilter
from csagent.retrieval.retriever import HybridRetriever

_FALLBACK = "죄송해요, 잘 못 들었어요. 한 번만 다시 말씀해 주시겠어요?"
_DISPOSITIONS = {"send_app_link": "link_sent", "transfer_to_agent": "transferred"}


class BaselineAgent:
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

    @property
    def name(self) -> str:
        return "A1" if self.gate else "A0"

    def respond(self, state: CallState, turn: TurnInput) -> AgentReply:
        for event in self.respond_stream(state, turn):
            if event.type == "done":
                return event.data["reply"]
        raise RuntimeError("respond_stream이 done 없이 끝났다")

    def respond_stream(self, state: CallState, turn: TurnInput) -> Iterator[AgentEvent]:
        t0 = time.perf_counter()
        state.turn += 1
        utterance = turn.utterance
        state.trace.add(state.turn, "turn_start", {"utterance": utterance, "event": turn.event})

        hits = self.retriever.search(
            self._query(state, utterance), self.k,
            flt=SearchFilter(audience=state.ctx.audience, companies=state.ctx.companies))
        t_retrieve = (time.perf_counter() - t0) * 1000
        state.trace.add(state.turn, "retrieval", {"chunk_ids": [h.chunk_id for h in hits]})
        yield AgentEvent("retrieval", {
            "ms": round(t_retrieve),
            "docs": [{"chunk_id": h.chunk_id, "title": h.title, "section": h.section,
                      "score": round(h.score, 4)} for h in hits]})

        messages = self._messages(state, utterance, hits)
        reply = AgentReply(text="", retrieved=[h.chunk_id for h in hits])
        said: list[str] = []
        llm_ms, first_token_ms, tokens = 0.0, None, {"prompt": 0, "completion": 0}

        for _ in range(self.max_steps):
            extractor, result = SayExtractor(), None
            for item in self.llm.stream_chat(messages, max_tokens=300, json_mode=True):
                if isinstance(item, LLMResult):
                    result = item
                    continue
                piece = extractor.feed(item)
                if piece:
                    first_token_ms = first_token_ms or (time.perf_counter() - t0) * 1000
                    yield AgentEvent("delta", {"text": piece})
            llm_ms += result.latency_ms
            tokens["prompt"] += result.prompt_tokens
            tokens["completion"] += result.completion_tokens
            state.trace.add(state.turn, "llm", {"role": "agent", "ms": round(result.latency_ms),
                                                 "prompt_tokens": result.prompt_tokens,
                                                 "completion_tokens": result.completion_tokens})
            try:
                out = parse_json(result.text)
            except (ValueError, json.JSONDecodeError):
                yield AgentEvent("delta", {"text": _FALLBACK})
                said.append(_FALLBACK)
                break

            self._apply_pending_answer(state, out.get("pending_answer"), utterance)
            say, tool = (out.get("say") or "").strip(), out.get("tool")
            if say:
                said.append(say)
            if not tool:
                break

            call, feedback, stop = self._run_tool(state, tool)
            reply.tool_calls.append(call)
            yield AgentEvent("tool_call", {"name": call.name, "args": call.args})
            yield AgentEvent("tool_result", {"name": call.name, "result": call.result})
            self._apply_disposition(reply, call)
            if stop:
                break
            messages += [{"role": "assistant", "content": result.text},
                         {"role": "user", "content": feedback}]

        reply.text = " ".join(said) or _FALLBACK
        if not said:
            yield AgentEvent("delta", {"text": _FALLBACK})
        state.history += [("customer", utterance), ("agent", reply.text)]
        reply.timings_ms = {"retrieve": t_retrieve, "llm": llm_ms, "first_token": first_token_ms or 0.0,
                            "total": (time.perf_counter() - t0) * 1000}
        reply.tokens = tokens
        state.trace.add(state.turn, "reply", {"text": reply.text, "disposition": reply.disposition})
        yield AgentEvent("done", {"reply": reply})

    # -- 도구 실행 --

    def _run_tool(self, state: CallState, tool: dict) -> tuple[ToolCall, str, bool]:
        """(호출 기록, LLM에게 돌려줄 메시지, 이번 턴을 끝낼지)."""
        name, args = tool.get("name", ""), tool.get("args") or {}
        call = ToolCall(name, args)
        state.trace.add(state.turn, "tool_request", {"name": name, "args": args})
        spec = self.tools.get(name)
        if spec is None:
            call.result = {"ok": False, "error": f"unknown_tool:{name}"}
            return call, f"[도구 오류] 없는 도구입니다: {name}", False

        if self.gate is not None:
            decision = self.gate.check(state, spec, args)
            state.trace.add(state.turn, "gate", {"name": name, "allowed": decision.allowed,
                                                 "reason": decision.reason})
            if not decision.allowed:
                call.result = {"ok": False, "blocked": decision.reason}
                return call, f"[실행 차단] {decision.message}", False

        outcome = self.tools.run(state, name, args)
        call.result = {"ok": outcome.ok, **outcome.result,
                       **({"error": outcome.error} if outcome.error else {})}
        state.trace.add(state.turn, "tool_result", {"name": name, "risk": spec.risk, **call.result})
        prefix = "[도구 결과]" if outcome.ok else "[도구 실패]"
        return call, f"{prefix} " + json.dumps(call.result, ensure_ascii=False), spec.risk == "handoff"

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

    @staticmethod
    def _query(state: CallState, utterance: str) -> str:
        """검색 질의 = 직전 고객 발화 + 이번 발화 (「그거요」 같은 짧은 후속 발화 보완)."""
        prev = [text for role, text in state.history if role == "customer"][-1:]
        return " ".join(prev + [utterance])

    def _messages(self, state: CallState, utterance: str, hits) -> list[dict]:
        docs = "\n\n".join(f"[{i}] {h.display_text()[:self.doc_chars]}"
                           for i, h in enumerate(hits, 1)) or "(없음)"
        p = state.pending
        pending = (f"{'실행 재확인' if p.kind == 'confirm' else '담당자 확인'}: "
                   f"{p.action} {json.dumps(p.target, ensure_ascii=False)}" if p else "(없음)")
        system = prompts.SYSTEM.format(tools=self.tools.prompt(), pending=pending)
        context = prompts.CONTEXT.format(
            today=state.ctx.today or "-", audience=state.ctx.audience,
            name=state.ctx.customer.get("name", "-"), docs=docs,
            auth="완료" if state.authenticated else "안 됨")
        # Qwen 채팅 템플릿은 system 메시지를 맨 앞 하나만 허용한다 → 규칙과 통화 정보를 합친다
        msgs = [{"role": "system", "content": system + "\n\n" + context}]
        for role, text in state.history:
            msgs.append({"role": "user" if role == "customer" else "assistant",
                         "content": text if role == "customer"
                         else json.dumps({"say": text}, ensure_ascii=False)})
        msgs.append({"role": "user", "content": utterance})
        return msgs


def reply_to_dict(reply: AgentReply) -> dict:
    """AgentReply → JSON 직렬화 가능한 dict (서버·결과 저장용)."""
    return asdict(reply)
