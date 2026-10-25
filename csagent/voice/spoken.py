"""TTS에 넣기 전 글자 → 읽는 말. 화면·기록의 글자는 그대로 두고 소리로만 바꾼다.

번호처럼 쓰이는 숫자는 한 자리씩 한글로 읽는다 (Supertonic이 「1234」를 「이릭 34」, 「1644-2944」를 「1644-인국444」로 읽었다 — 10/25 시연 녹화).
- 전화번호 1644-2944 → 일육사사, 이구사사
- 영문+숫자 번호 A3001 → 에이 삼공공일
- 단위 없는 세 자리 이상 숫자 1234가 → 일이삼사가 (카드 끝자리·주문번호·인증번호)
단위가 붙은 수(30만 원, 10월 12일, 2영업일, 245mm, 2026년)와 두 자리 이하 숫자는 그대로 둔다 — TTS가 수로 바르게 읽는다.
"""

from __future__ import annotations

import re

_DIGIT = dict(zip("0123456789", ["공", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]))
_LETTER = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                   ["에이", "비", "씨", "디", "이", "에프", "지", "에이치", "아이", "제이", "케이", "엘", "엠", "엔", "오", "피",
                    "큐", "알", "에스", "티", "유", "브이", "더블유", "엑스", "와이", "지"]))
# 수 뒤에 오면 「수」로 읽는 단위 (번호가 아니다)
_UNIT = (r"(?:\s*(?:원|만|천|억|조|년|월|일|시간|시|분|초|개월|개|명|건|회|번째|번|차|자리|점|%|퍼센트|mm|cm|km|kg|g|ml|"
         r"영업일|주|살|배|장|대|층|호|kb|mb|gb))")
_PHONE = re.compile(r"(?<![\d-])\d{2,4}(?:-\d{3,4}){1,2}(?![\d-])")
_ALNUM = re.compile(r"(?<![A-Za-z\d])([A-Za-z]{1,3})(\d{2,})(?![\d])")
_RUN = re.compile(rf"(?<![\d,.])(\d{{3,}})(?![\d,.])(?!{_UNIT})")


def _digits(s: str) -> str:
    return "".join(_DIGIT[c] for c in s)


def for_tts(text: str) -> str:
    text = _PHONE.sub(lambda m: ", ".join(_digits(g) for g in m.group(0).split("-")), text)
    text = _ALNUM.sub(lambda m: " ".join(_LETTER.get(c.upper(), c) for c in m.group(1)) + " " + _digits(m.group(2)), text)
    return _RUN.sub(lambda m: _digits(m.group(1)), text)
