"""스트리밍 응답(delta)을 TTS로 보낼 문장 단위로 끊는다.

TTS(Supertonic)는 진짜 스트리밍이 아니라 문장 하나를 통째로 합성한다 → 첫 소리 지연 = 첫 문장이 완성될 때까지
+ 그 문장 합성 시간. 그래서 문장 끝(. ? !)이 오면 바로 내보내고, 끝 표시 없이 너무 길어지면 쉼표·공백에서 끊는다.
「1.5」 같은 소수점에서 끊지 않도록 마침표 다음 글자가 공백이거나 응답이 끝났을 때만 문장 끝으로 본다.
"""

from __future__ import annotations

_ENDS = ".?!。…"


class SentenceSplitter:
    def __init__(self, max_chars: int = 80) -> None:
        self.max_chars = max_chars
        self._buf = ""

    def feed(self, text: str) -> list[str]:
        self._buf += text
        out = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            piece, self._buf = self._buf[:cut].strip(), self._buf[cut:].lstrip()
            if piece:
                out.append(piece)
        return out

    def flush(self) -> list[str]:
        piece, self._buf = self._buf.strip(), ""
        return [piece] if piece else []

    def _find_cut(self):
        b = self._buf
        for i, ch in enumerate(b):
            if ch in _ENDS and i + 1 < len(b) and (b[i + 1].isspace() or b[i + 1] in _ENDS + "\"')"):
                j = i + 1
                while j < len(b) and b[j] in _ENDS:
                    j += 1
                return j
            if ch == "\n":
                return i + 1
        if len(b) > self.max_chars:   # 끝 표시 없이 길어짐 → 마지막 쉼표, 없으면 마지막 공백에서
            head = b[:self.max_chars]
            k = max(head.rfind(","), head.rfind("，"))
            if k < 0:
                k = head.rfind(" ")
            return (k + 1) if k > 0 else self.max_chars
        return None
