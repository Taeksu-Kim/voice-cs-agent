"""LLM 클라이언트 — 에이전트·고객 시뮬레이터·채점기가 함께 쓴다.

OpenAI 호환 /chat/completions (vLLM). Qwen3 계열은 기본으로 생각(thinking) 토큰을 길게 내므로
음성 지연 예산을 위해 기본은 끈다 (enable_thinking=False).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Protocol, Sequence, Union


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    first_token_ms: float = 0.0   # 스트리밍일 때 첫 토큰까지 (음성 응답 시작 지연)
    call_id: str = ""             # 입출력 기록(llm_log)과 trace를 잇는 id. 기록하지 않으면 빈 값


# 스트리밍은 텍스트 조각(str)들을 내보내고, 마지막에 전체 결과(LLMResult)를 한 번 내보낸다
StreamItem = Union[str, LLMResult]


class ChatLLM(Protocol):
    def chat(self, messages: Sequence[dict], *, max_tokens: int = 512, temperature: float = 0.0,
             json_mode: bool = False, schema: Optional[dict] = None) -> LLMResult: ...

    def stream_chat(self, messages: Sequence[dict], *, max_tokens: int = 512, temperature: float = 0.0,
                    json_mode: bool = False, schema: Optional[dict] = None) -> Iterator[StreamItem]: ...


def _transient(e: Exception) -> bool:
    """다시 보내면 될 수도 있는 실패인가 — 타임아웃·연결 끊김·서버 5xx (4xx는 요청 자체가 틀린 것)."""
    import requests  # lazy

    if isinstance(e, (requests.Timeout, requests.ConnectionError)):
        return True
    return isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code >= 500


class OpenAICompatLLM:
    def __init__(self, base_url: str, model: str, *, thinking: bool = False,
                 timeout: float = 120.0, retries: int = 2, backoff_s: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.thinking = thinking
        self.timeout = timeout
        self.retries = retries        # 일시적 실패만 다시 보낸다 (_transient)
        self.backoff_s = backoff_s    # n번째 재시도 전 backoff_s * n초 쉰다

    def chat(self, messages: Sequence[dict], *, max_tokens: int = 512, temperature: float = 0.0,
             json_mode: bool = False, schema: Optional[dict] = None) -> LLMResult:
        import requests  # lazy

        body = self._body(messages, max_tokens, temperature, json_mode, schema)
        t0 = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout)
                r.raise_for_status()
                break
            except Exception as e:
                if attempt == self.retries or not _transient(e):
                    raise
                time.sleep(self.backoff_s * (attempt + 1))
        data = r.json()
        usage = data.get("usage") or {}
        return LLMResult(
            text=data["choices"][0]["message"]["content"] or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def stream_chat(self, messages: Sequence[dict], *, max_tokens: int = 512, temperature: float = 0.0,
                    json_mode: bool = False, schema: Optional[dict] = None) -> Iterator[StreamItem]:
        """토큰 단위 스트리밍. 소비자가 중간에 멈추면(generator close) 연결을 끊어 생성도 멈춘다.
        재시도는 첫 조각을 내보내기 전에 실패했을 때만 한다 (이미 내보낸 말은 되돌릴 수 없다)."""
        body = {**self._body(messages, max_tokens, temperature, json_mode, schema),
                "stream": True, "stream_options": {"include_usage": True}}
        t0 = time.perf_counter()
        first_ms, parts, usage = None, [], {}
        for attempt in range(self.retries + 1):
            try:
                for data in self._stream_events(body):
                    usage = data.get("usage") or usage
                    for choice in data.get("choices", []):
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            first_ms = first_ms or (time.perf_counter() - t0) * 1000
                            parts.append(piece)
                            yield piece
                break
            except Exception as e:
                if parts or attempt == self.retries or not _transient(e):
                    raise
                time.sleep(self.backoff_s * (attempt + 1))
        yield LLMResult(text="".join(parts), prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        first_token_ms=first_ms or 0.0)

    def _stream_events(self, body: dict) -> Iterator[dict]:
        import requests  # lazy

        with requests.post(f"{self.base_url}/chat/completions", json=body,
                           timeout=self.timeout, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith("data: ") and line != "data: [DONE]":
                    yield json.loads(line[6:])

    def _body(self, messages, max_tokens, temperature, json_mode, schema=None) -> dict:
        body = {
            "model": self.model, "messages": list(messages), "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        if schema:   # 키·값 형식까지 강제 (vLLM 구조화 출력). 키 순서는 schema의 properties 순서를 따른다
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "output", "schema": schema}}
        elif json_mode:
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
