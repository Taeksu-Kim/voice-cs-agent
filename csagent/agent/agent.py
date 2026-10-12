"""상담 에이전트 — 한 턴: 검색 → LLM(JSON, 스트리밍) → (도구 실행 → LLM) → 응답.

본체는 respond_stream(): 단계마다 이벤트를 내보낸다 (웹 서버 SSE가 그대로 전달).
respond()는 그 스트림을 끝까지 소비해 최종 AgentReply만 돌려준다 (평가 파이프라인용).

상태는 밖(대화 이력·CallContext)에 두고, 에이전트는 매 턴 그것을 받아 응답만 만든다.
그래야 평가 파이프라인이 같은 조건으로 반복 실행·재현할 수 있다.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Iterator, Sequence

from csagent.agent import prompts
from csagent.agent.contracts import AgentEvent, AgentReply, CallContext, ToolCall, TurnInput
from csagent.agent.streaming import SayExtractor
from csagent.agent.tools import TOOL_SPECS, ToolBox
from csagent.llm import ChatLLM, LLMResult, parse_json
from csagent.retrieval.filters import SearchFilter
from csagent.retrieval.retriever import HybridRetriever

# 결과를 받아 다시 말해야 하는 도구 vs 실행하고 바로 말하는 도구
_LOOKUP_TOOLS = {"lookup_orders"}
_FALLBACK = "죄송해요, 잘 못 들었어요. 한 번만 다시 말씀해 주시겠어요?"


class CSAgent:
    def __init__(self, retriever: HybridRetriever, llm: ChatLLM, *, k: int = 4,
                 doc_chars: int = 700, max_steps: int = 3) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.doc_chars = doc_chars
        self.max_steps = max_steps

    def respond(self, ctx: CallContext, history: Sequence[tuple[str, str]],
                turn: TurnInput) -> AgentReply:
        for event in self.respond_stream(ctx, history, turn):
            if event.type == "done":
                return event.data["reply"]
        raise RuntimeError("respond_stream이 done 없이 끝났다")

    def respond_stream(self, ctx: CallContext, history: Sequence[tuple[str, str]],
                       turn: TurnInput) -> Iterator[AgentEvent]:
        """history: [("customer"|"agent", 발화)] — 이번 고객 발화는 turn으로 따로 받는다."""
        t0 = time.perf_counter()
        hits = self.retriever.search(
            self._query(history, turn), self.k,
            flt=SearchFilter(audience=ctx.audience, companies=ctx.companies))
        t_retrieve = (time.perf_counter() - t0) * 1000
        yield AgentEvent("retrieval", {
            "ms": round(t_retrieve),
            "docs": [{"chunk_id": h.chunk_id, "title": h.title, "section": h.section,
                      "score": round(h.score, 4)} for h in hits]})

        messages = self._messages(ctx, history, turn, hits)
        tools, reply = ToolBox(ctx), AgentReply(text="", retrieved=[h.chunk_id for h in hits])
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
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - t0) * 1000
                    yield AgentEvent("delta", {"text": piece})
            llm_ms += result.latency_ms
            tokens["prompt"] += result.prompt_tokens
            tokens["completion"] += result.completion_tokens
            try:
                out = parse_json(result.text)
            except (ValueError, json.JSONDecodeError):
                yield AgentEvent("delta", {"text": _FALLBACK})
                said.append(_FALLBACK)
                break
            say, tool = (out.get("say") or "").strip(), out.get("tool")
            if say:
                said.append(say)  # 도구 호출 전 「확인해 볼게요」 같은 말도 음성으론 그대로 나간다
            if not tool:
                break
            call = ToolCall(tool.get("name", ""), tool.get("args") or {})
            yield AgentEvent("tool_call", {"name": call.name, "args": call.args})
            call.result = tools.call(call.name, call.args)
            yield AgentEvent("tool_result", {"name": call.name, "result": call.result})
            reply.tool_calls.append(call)
            if call.name not in _LOOKUP_TOOLS:
                self._apply_action(reply, call)
                break
            messages += [{"role": "assistant", "content": result.text},
                         {"role": "user", "content": "[도구 결과] " +
                          json.dumps(call.result, ensure_ascii=False)}]

        reply.text = " ".join(said) or _FALLBACK
        if not said:
            yield AgentEvent("delta", {"text": _FALLBACK})
        reply.timings_ms = {"retrieve": t_retrieve, "llm": llm_ms,
                            "first_token": first_token_ms or 0.0,
                            "total": (time.perf_counter() - t0) * 1000}
        reply.tokens = tokens
        yield AgentEvent("done", {"reply": reply})

    # -- 내부 --

    @staticmethod
    def _query(history: Sequence[tuple[str, str]], turn: TurnInput) -> str:
        """검색 질의 = 직전 고객 발화 + 이번 발화 (「그거요」 같은 짧은 후속 발화 보완)."""
        prev = [text for role, text in history if role == "customer"][-1:]
        return " ".join(prev + [turn.utterance])

    def _messages(self, ctx: CallContext, history, turn: TurnInput, hits) -> list[dict]:
        docs = "\n\n".join(f"[{i}] {h.display_text()[:self.doc_chars]}"
                           for i, h in enumerate(hits, 1)) or "(없음)"
        context = prompts.CONTEXT.format(today=ctx.today or "-", audience=ctx.audience,
                                         name=ctx.customer.get("name", "-"), docs=docs)
        # Qwen 채팅 템플릿은 system 메시지를 맨 앞 하나만 허용한다 → 규칙과 통화 정보를 합친다
        msgs = [{"role": "system",
                 "content": prompts.SYSTEM.format(tools=TOOL_SPECS) + "\n\n" + context}]
        for role, text in history:
            msgs.append({"role": "user" if role == "customer" else "assistant",
                         "content": text if role == "customer" else json.dumps({"say": text}, ensure_ascii=False)})
        msgs.append({"role": "user", "content": turn.utterance})
        return msgs

    @staticmethod
    def _apply_action(reply: AgentReply, call: ToolCall) -> None:
        if call.name == "end_call":
            reply.end_call = True
            reply.disposition = call.args.get("disposition", "resolved")
        elif call.name == "transfer_to_agent":
            reply.end_call = True
            reply.disposition = "transferred"
        elif call.name == "send_app_link":
            reply.disposition = "link_sent"


def reply_to_dict(reply: AgentReply) -> dict:
    """AgentReply → JSON 직렬화 가능한 dict (서버·결과 저장용)."""
    return asdict(reply)
