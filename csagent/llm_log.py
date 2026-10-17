"""LLM 입출력 기록 — 호출마다 보낸 메시지 전체와 받은 원문을 JSONL 한 줄로 남긴다.

추적 로그(trace)는 판단 흐름만 담고 프롬프트·원문은 담지 않는다. 둘은 call_id로 잇는다:
trace의 llm·route 이벤트에 call_id가 있고, 이 파일에서 같은 id를 찾으면 그 호출의 입력·출력이 나온다.

    llm = LoggedLLM(OpenAICompatLLM(...), LLMCallLog(path), source="agent")
    with log_context(scenario="s4_b5_fraud", repeat=0): ...   # 이 안의 호출에 붙는 꼬리표 (csagent.obs)

source: agent(워커 루프) | router | customer(시뮬레이터) | judge
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from csagent.llm import ChatLLM, LLMResult, StreamItem
from csagent.obs import current_context, event, log_context  # noqa: F401  (log_context는 여기서도 꺼내 쓴다)

log = logging.getLogger("csagent.llm")


class LLMCallLog:
    """JSONL 파일 하나. 여러 LoggedLLM(출처별)이 같은 파일을 쓴다."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class LoggedLLM:
    """ChatLLM을 감싸 호출마다 기록하고, 결과에 call_id를 붙인다. 동작은 바꾸지 않는다."""

    def __init__(self, inner: ChatLLM, log: LLMCallLog, source: str) -> None:
        self.inner = inner
        self.log = log
        self.source = source

    def with_source(self, source: str) -> "LoggedLLM":
        return LoggedLLM(self.inner, self.log, source)

    def chat(self, messages: Sequence[dict], **kw) -> LLMResult:
        call_id = _new_id()
        try:
            result = self.inner.chat(messages, **kw)
        except Exception as e:
            self._write(call_id, messages, kw, None, error=repr(e))
            raise
        result.call_id = call_id
        self._write(call_id, messages, kw, result)
        return result

    def stream_chat(self, messages: Sequence[dict], **kw) -> Iterator[StreamItem]:
        call_id, parts, result, error = _new_id(), [], None, None
        try:
            for item in self.inner.stream_chat(messages, **kw):
                if isinstance(item, LLMResult):
                    result = item
                    result.call_id = call_id
                else:
                    parts.append(item)
                yield item
        except GeneratorExit:
            error = "aborted"   # 소비자가 중간에 멈춤 — 받은 데까지 남긴다
            raise
        except Exception as e:
            error = repr(e)
            raise
        finally:
            self._write(call_id, messages, kw, result, partial="".join(parts), error=error)

    def _write(self, call_id: str, messages: Sequence[dict], kw: dict, result: Optional[LLMResult],
               partial: str = "", error: Optional[str] = None) -> None:
        record = {"call_id": call_id, "ts": time.time(), "source": self.source, **current_context(),
                  "params": kw, "messages": list(messages),
                  "output": result.text if result else partial}
        if result:
            record.update(prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                          ms=round(result.latency_ms))
        if error:
            record["error"] = error
            if error != "aborted":   # 중단은 끼어들기 등 정상 흐름
                event(log, "llm_error", logging.ERROR, llm_call_id=call_id, source=self.source, error=error)
        self.log.write(record)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]
