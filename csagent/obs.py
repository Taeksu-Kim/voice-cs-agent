"""관측(로그) 공통 — 운영 이벤트 로그 설정과 이벤트 기록 함수.

기록은 세 갈래로 나눈다. 모두 같은 폴더(로그 폴더)에 JSONL로 남는다.
- events.jsonl     운영 이벤트 (이 모듈). 턴 지표·폴백·차단·오류 등 — 모니터링 집계의 원천
- llm_calls.jsonl  LLM 입출력 원문 (llm_log.py). 크기가 커서 따로 둔다
- calls.jsonl      통화가 끝날 때 통화 1건의 trace 전체 (서버). 평가는 results.jsonl에 들어간다
로그 폴더: 평가는 runs/<시각>/, 서버는 logs/<날짜>/.

이벤트 한 줄: {"ts", "level", "event", "logger", <맥락: scenario·repeat 등>, <필드: call_id·turn 등>}
같은 통화의 이벤트·trace·LLM 원문은 call_id(통화)와 llm_call_id(LLM 호출)로 잇는다.

    setup_logging(run_dir)                       # 프로세스 시작 시 한 번
    with log_context(scenario="s4_b5_fraud"):    # 이 안의 이벤트·LLM 기록에 꼬리표가 붙는다
        event(log, "fallback", logging.WARNING, call_id=..., reason="parse_error")
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

EVENTS_FILE = "events.jsonl"
LLM_CALLS_FILE = "llm_calls.jsonl"
CALLS_FILE = "calls.jsonl"

_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("log_context", default={})


@contextlib.contextmanager
def log_context(**tags: Any) -> Iterator[None]:
    token = _context.set({**_context.get(), **tags})
    try:
        yield
    finally:
        _context.reset(token)


def current_context() -> dict[str, Any]:
    return _context.get()


def event(logger: logging.Logger, name: str, level: int = logging.INFO, **fields: Any) -> None:
    """이벤트 한 건. name이 집계 키이고, fields는 그대로 JSON 필드가 된다."""
    logger.log(level, name, extra={"event": name, "fields": fields})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": round(record.created, 3), "level": record.levelname,
               "event": getattr(record, "event", None) or "log", "logger": record.name,
               **current_context(), **getattr(record, "fields", {})}
        if not getattr(record, "event", None):
            out["msg"] = record.getMessage()
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """사람이 읽는 한 줄 (stderr). 필드는 key=value로."""

    def format(self, record: logging.LogRecord) -> str:
        fields = {**current_context(), **getattr(record, "fields", {})}
        kv = " ".join(f"{k}={_short(v)}" for k, v in fields.items())
        line = f"{time.strftime('%H:%M:%S', time.localtime(record.created))} {record.levelname:<7} " \
               f"{record.getMessage()} {kv}".rstrip()
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(log_dir: Optional[Path], *, console_level: int = logging.WARNING) -> Optional[Path]:
    """csagent 로거에 파일(JSON, INFO 이상)·콘솔(console_level 이상) 핸들러를 단다. 다시 부르면 교체한다."""
    root = logging.getLogger("csagent")
    root.setLevel(logging.INFO)
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)
    if log_dir is None:
        return None
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file = _FlushingFileHandler(log_dir / EVENTS_FILE, encoding="utf-8")
    file.setLevel(logging.INFO)
    file.setFormatter(JsonFormatter())
    root.addHandler(file)
    return log_dir


class _FlushingFileHandler(logging.FileHandler):
    """기록마다 flush — 실행 중에도 모니터가 바로 읽을 수 있게."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def _short(v: Any, limit: int = 80) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…"
