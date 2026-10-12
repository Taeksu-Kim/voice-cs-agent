"""LLM 클라이언트 — 에이전트·고객 시뮬레이터·채점기가 함께 쓴다.

OpenAI 호환 /chat/completions (vLLM). Qwen3 계열은 기본으로 생각(thinking) 토큰을 길게 내므로
음성 지연 예산을 위해 기본은 끈다 (enable_thinking=False).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence, Union


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    first_token_ms: float = 0.0   # 스트리밍일 때 첫 토큰까지 (음성 응답 시작 지연)


# 스트리밍은 텍스트 조각(str)들을 내보내고, 마지막에 전체 결과(LLMResult)를 한 번 내보낸다
StreamItem = Union[str, LLMResult]


class ChatLLM(Protocol):
    def chat(self, messages: Sequence[dict], *, max_tokens: int = 512,
             temperature: float = 0.0, json_mode: bool = False) -> LLMResult: ...

    def stream_chat(self, messages: Sequence[dict], *, max_tokens: int = 512,
                    temperature: float = 0.0, json_mode: bool = False) -> Iterator[StreamItem]: ...


class OpenAICompatLLM:
    def __init__(self, base_url: str, model: str, *, thinking: bool = False,
                 timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.thinking = thinking
        self.timeout = timeout

    def chat(self, messages: Sequence[dict], *, max_tokens: int = 512,
             temperature: float = 0.0, json_mode: bool = False) -> LLMResult:
        import requests  # lazy

        body = self._body(messages, max_tokens, temperature, json_mode)
        t0 = time.perf_counter()
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        return LLMResult(
            text=data["choices"][0]["message"]["content"] or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def stream_chat(self, messages: Sequence[dict], *, max_tokens: int = 512,
                    temperature: float = 0.0, json_mode: bool = False) -> Iterator[StreamItem]:
        """토큰 단위 스트리밍. 소비자가 중간에 멈추면(generator close) 연결을 끊어 생성도 멈춘다."""
        import requests  # lazy

        body = {**self._body(messages, max_tokens, temperature, json_mode),
                "stream": True, "stream_options": {"include_usage": True}}
        t0 = time.perf_counter()
        first_ms, parts, usage = None, [], {}
        with requests.post(f"{self.base_url}/chat/completions", json=body,
                           timeout=self.timeout, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                data = json.loads(line[6:])
                usage = data.get("usage") or usage
                for choice in data.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        first_ms = first_ms or (time.perf_counter() - t0) * 1000
                        parts.append(piece)
                        yield piece
        yield LLMResult(text="".join(parts), prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        first_token_ms=first_ms or 0.0)

    def _body(self, messages, max_tokens, temperature, json_mode) -> dict:
        body = {
            "model": self.model, "messages": list(messages), "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body


class ScriptedLLM:
    """미리 정한 응답을 순서대로 돌려준다 (테스트 전용). 받은 메시지는 calls에 남는다."""

    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[dict]] = []

    def chat(self, messages: Sequence[dict], **_) -> LLMResult:
        self.calls.append(list(messages))
        return LLMResult(text=self.outputs.pop(0))

    def stream_chat(self, messages: Sequence[dict], **_) -> Iterator[StreamItem]:
        result = self.chat(messages)
        for i in range(0, len(result.text), 3):  # 3글자씩 흘려보낸다
            yield result.text[i:i + 3]
        yield result


def parse_json(text: str) -> dict:
    """LLM 출력에서 JSON 객체를 꺼낸다 (코드펜스·앞뒤 잡음 허용)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))
