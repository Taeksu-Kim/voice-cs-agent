"""발화 구간 판정 — Silero VAD(ONNX) 스트리밍 + 발화 시작·끝 판정(endpointing).

브라우저가 보내는 16kHz PCM을 512샘플(32ms) 단위로 Silero에 넣어 발화 확률을 얻고,
확률이 일정 시간 이상 이어지면 「발화 시작」, 침묵이 일정 시간 이상 이어지면 「발화 끝」으로 본다.
- 시작을 바로 내지 않고 min_speech_ms 동안 이어질 때 낸다: 기침·키보드 소리로 끼어들기가 걸리지 않게
- 끝은 침묵 min_silence_ms 뒤에 낸다: 이 값이 그대로 응답 지연에 더해진다 (지연 예산 항목)

모델 파일: faster-whisper 1.2.1에 들어 있는 silero_vad_v6.onnx (MIT) 를 복사해 둔 것.
입력 형식(앞 청크 끝 64샘플 + 이번 512샘플, 상태 h·c)은 faster_whisper/vad.py의 SileroVADModel을 따랐다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

SR = 16000
CHUNK = 512           # Silero 16kHz 입력 단위 (32ms)
CONTEXT = 64
MODEL_PATH = Path(__file__).with_name("silero_vad_v6.onnx")


class SileroVAD:
    """청크 하나씩 넣어 발화 확률을 받는다. 통화마다 하나 (RNN 상태를 가진다)."""

    def __init__(self, path: Path = MODEL_PATH) -> None:
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._sess = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"],
                                                  sess_options=opts)
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._ctx = np.zeros(CONTEXT, dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        x = np.concatenate([self._ctx, chunk.astype(np.float32)])[None, :]
        prob, self._h, self._c = self._sess.run(None, {"input": x, "h": self._h, "c": self._c})
        self._ctx = chunk[-CONTEXT:].astype(np.float32)
        return float(np.asarray(prob).reshape(-1)[0])


@dataclass
class VadEvent:
    kind: str                         # "start" | "end"
    t: float                          # 스트림 시작부터의 시각(초) — 시작은 발화가 실제로 시작된 시점, 끝은 판정 시점
    audio: Optional[np.ndarray] = None   # end일 때 발화 구간 PCM(float32, 앞뒤 여유 포함)
    speech_end_t: Optional[float] = None  # end일 때 마지막으로 말소리가 있던 시점 (판정 시점 - 침묵 대기)


class Endpointer:
    """PCM을 받아 발화 시작·끝 이벤트를 낸다. prob_fn은 테스트에서 가짜로 바꿔 끼운다."""

    def __init__(self, prob_fn=None, *, threshold: float = 0.5, neg_threshold: float = 0.35,
                 min_speech_ms: int = 250, min_silence_ms: int = 600, pre_roll_ms: int = 300) -> None:
        self.prob = prob_fn or SileroVAD()
        self.threshold, self.neg_threshold = threshold, neg_threshold
        self.min_speech = int(min_speech_ms / 1000 * SR / CHUNK)
        self.min_silence = int(min_silence_ms / 1000 * SR / CHUNK)
        self.pre_roll = int(pre_roll_ms / 1000 * SR / CHUNK)
        self._pending = np.zeros(0, dtype=np.float32)
        self._n = 0                # 처리한 청크 수
        self._chunks: list[np.ndarray] = []   # 발화 중이면 발화 구간, 아니면 최근 pre_roll 청크
        self._run = 0              # 연속 말소리 청크 수 (시작 판정용)
        self._quiet = 0            # 연속 침묵 청크 수 (끝 판정용)
        self.in_speech = False

    def feed(self, pcm: np.ndarray) -> list[VadEvent]:
        self._pending = np.concatenate([self._pending, pcm.astype(np.float32)])
        out = []
        while len(self._pending) >= CHUNK:
            chunk, self._pending = self._pending[:CHUNK], self._pending[CHUNK:]
            ev = self._step(chunk)
            if ev:
                out.append(ev)
        return out

    def _step(self, chunk: np.ndarray) -> Optional[VadEvent]:
        p = self.prob(chunk)
        self._n += 1
        self._chunks.append(chunk)
        t = self._n * CHUNK / SR
        if not self.in_speech:
            self._run = self._run + 1 if p >= self.threshold else 0
            keep = self.pre_roll + self._run
            if len(self._chunks) > keep:
                del self._chunks[:len(self._chunks) - keep]
            if self._run >= self.min_speech:
                self.in_speech, self._quiet = True, 0
                return VadEvent("start", t - self._run * CHUNK / SR)
            return None
        self._quiet = self._quiet + 1 if p < self.neg_threshold else 0
        if self._quiet >= self.min_silence:
            audio = np.concatenate(self._chunks)
            self.in_speech, self._run, self._chunks = False, 0, []
            return VadEvent("end", t, audio=audio, speech_end_t=t - self._quiet * CHUNK / SR)
        return None
