"""JSON 출력 스트림에서 "say" 값만 실시간으로 꺼내는 파서.

에이전트는 {"intent": ..., ..., "say": "...", "tool": ...} JSON을 출력한다. 전체 JSON이 끝날 때까지
기다리면 음성 응답 시작이 늦어지므로, 생성 중인 문자열에서 say 값 부분만 글자 단위로 흘려보낸다.
intent는 say보다 앞에 오므로, say가 흐르기 전에 「다른 업무로 전환(switch)」인지 알 수 있다.
(우리가 정한 출력 형식을 읽는 것 — 말의 의미를 판단하는 정규식이 아니다)
"""

from __future__ import annotations

import json
import re
from typing import Optional

_SAY_START = re.compile(r'"say"\s*:\s*"')
_INTENT = re.compile(r'"intent"\s*:\s*"(\w+)"')
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "n": "\n", "t": "\t", "r": "\r", "b": "", "f": ""}


class SayExtractor:
    def __init__(self) -> None:
        self._buf = ""
        self._pos = -1       # say 문자열 안에서 다음에 읽을 위치 (-1 = 아직 시작 전)
        self.done = False

    @property
    def intent(self) -> Optional[str]:
        m = _INTENT.search(self._buf)
        return m.group(1) if m else None

    def feed(self, chunk: str) -> str:
        """새로 들어온 조각을 받고, 새로 확정된 say 글자들을 돌려준다."""
        if self.done:
            return ""
        self._buf += chunk
        if self._pos < 0:
            m = _SAY_START.search(self._buf)
            if not m:
                return ""
            self._pos = m.end()
        out = []
        while self._pos < len(self._buf):
            ch = self._buf[self._pos]
            if ch == '"':
                self.done = True
                break
            if ch != "\\":
                out.append(ch)
                self._pos += 1
                continue
            # 이스케이프: 다음 글자(\uXXXX면 5글자)가 다 들어올 때까지 기다린다
            nxt = self._buf[self._pos + 1:self._pos + 2]
            if not nxt:
                break
            if nxt == "u":
                code = self._buf[self._pos + 2:self._pos + 6]
                if len(code) < 4:
                    break
                out.append(json.loads(f'"\\u{code}"'))
                self._pos += 6
            else:
                out.append(_ESCAPES.get(nxt, nxt))
                self._pos += 2
        return "".join(out)



# 발송·예약을 「~해 드릴게요」로 알린 뒤 도구가 성공하면 결과 스텝에서 「~해 드렸습니다」와 앞의 안내를 또 말해
# 길어졌다 — 지시문으로는 안 막혔다 (o1_h6, runs/20261018_075903_C). 완료 문장만 거르면 고객이 안 보낸 줄 알고
# 다시 요청해 두 번 보냈다 (s4_b0_stranger, runs/20261018_100842_C) → 결과 스텝 대신 짧은 완료 한마디를 코드가 붙인다.
# 「알림」은 표면 형태만 본다 (말의 의미 판단이 아니라 도구별로 정해 둔 표현).
_ANNOUNCED = {
    "send_app_link": (re.compile(r"(보내|발송해)\s?드릴|보낼게"), "보내 드렸어요."),
    "schedule_callback": (re.compile(r"예약해\s?드릴|예약할게"), "예약해 두었어요."),
}


def announced_done(tool: str, said: str) -> Optional[str]:
    """이번 턴에 이미 tool을 하겠다고 말했으면, 도구 성공 뒤 붙일 완료 한마디를 돌려준다."""
    pair = _ANNOUNCED.get(tool)
    return pair[1] if pair and pair[0].search(said) else None
