"""LLM 클라이언트 재시도 — 일시적 실패(타임아웃·5xx)만 다시 보내고, 스트리밍은 첫 조각 전까지만."""

import pytest
import requests

from csagent.llm import LLMResult, OpenAICompatLLM


class _Resp:
    def __init__(self, status=200, data=None, lines=()):
        self.status_code, self._data, self._lines = status, data, lines

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self._data

    def iter_lines(self, decode_unicode=True):
        for x in self._lines:
            if isinstance(x, Exception):
                raise x
            yield x

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_post(monkeypatch, outcomes):
    calls = []

    def post(*a, **kw):
        calls.append(1)
        o = outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o
    monkeypatch.setattr(requests, "post", post)
    return calls


OK = {"choices": [{"message": {"content": "네"}}], "usage": {}}


def test_chat_retries_timeout_then_succeeds(monkeypatch):
    calls = _fake_post(monkeypatch, [requests.Timeout(), _Resp(503), _Resp(data=OK)])
    llm = OpenAICompatLLM("http://x", "m", retries=2, backoff_s=0)
    assert llm.chat([{"role": "user", "content": "a"}]).text == "네"
    assert len(calls) == 3


def test_chat_gives_up_after_retries(monkeypatch):
    calls = _fake_post(monkeypatch, [requests.Timeout()] * 3)
    with pytest.raises(requests.Timeout):
        OpenAICompatLLM("http://x", "m", retries=2, backoff_s=0).chat([])
    assert len(calls) == 3


def test_chat_does_not_retry_client_error(monkeypatch):
    calls = _fake_post(monkeypatch, [_Resp(400)])
    with pytest.raises(requests.HTTPError):
        OpenAICompatLLM("http://x", "m", backoff_s=0).chat([])
    assert len(calls) == 1


def test_stream_retries_only_before_first_piece(monkeypatch):
    piece = 'data: {"choices": [{"delta": {"content": "안"}}]}'
    _fake_post(monkeypatch, [requests.ConnectionError(), _Resp(lines=[piece, "data: [DONE]"])])
    out = list(OpenAICompatLLM("http://x", "m", backoff_s=0).stream_chat([]))
    assert out[0] == "안" and isinstance(out[-1], LLMResult) and out[-1].text == "안"

    calls = _fake_post(monkeypatch, [_Resp(lines=[piece, requests.ConnectionError()]), _Resp(lines=[piece])])
    with pytest.raises(requests.ConnectionError):
        list(OpenAICompatLLM("http://x", "m", backoff_s=0).stream_chat([]))
    assert len(calls) == 1   # 이미 말을 내보냈으면 다시 보내지 않는다
