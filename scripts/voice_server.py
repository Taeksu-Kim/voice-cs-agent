"""음성 서버 — STT(Whisper large-v3-turbo int8, GPU)와 TTS(Supertonic 3, CPU)를 HTTP로 제공한다.

    ~/miniconda3/envs/voice-eval/bin/python scripts/voice_server.py [--port 8013]

    GET  /health
    POST /stt?degrade=tel   본문 = PCM16 little-endian 16kHz mono 원시 바이트 → {text, ms, audio_s}
                            degrade=tel이면 전사 전에 8kHz 전화 대역으로 열화 (데모·강건성 확인용)
    POST /tts               {text, voice: "F1", steps: 8} → audio/wav (44.1kHz), 헤더 X-Gen-Ms · X-Audio-S

배치 근거 (2026-10-14 실측, HANDOFF): STT+Qwen3-TTS를 둘 다 GPU에 두면 VRAM이 넘친다 →
TTS는 CPU 전용 Supertonic(RTF 0.14~0.21, VRAM 0). Whisper는 GPU 약 1.55GB(CUDA 컨텍스트 포함).
웹 서버(env customer-service-agent)와 env가 달라 별도 프로세스로 띄운다.
"""
from __future__ import annotations

import argparse
import io
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_bench import to_telephone  # noqa: E402

VOICES = [f"{g}{i}" for g in "MF" for i in range(1, 6)]


class TTSRequest(BaseModel):
    text: str
    voice: str = "F1"
    steps: int = 8


def create_app(threads: int | None = None) -> FastAPI:
    import torch  # noqa: F401 — 먼저 올려야 torch가 딸려 온 libcublas.so.12를 ctranslate2가 찾는다
    from faster_whisper import WhisperModel
    from supertonic import TTS

    stt = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
    tts = TTS(model="supertonic-3", intra_op_num_threads=threads)
    styles = {v: tts.get_voice_style(v) for v in VOICES}
    # 모델마다 락 하나 — STT와 TTS는 서로 다른 장치라 동시에 돈다 (에이전트 말하는 중 고객 발화 전사)
    stt_lock, tts_lock = threading.Lock(), threading.Lock()

    def transcribe(pcm: np.ndarray) -> str:
        segs, _ = stt.transcribe(pcm, language="ko", beam_size=1, condition_on_previous_text=False,
                                 vad_filter=False)   # 발화 구간은 이미 웹 서버의 VAD가 잘라 보낸다
        return "".join(s.text for s in segs).strip()

    # 첫 요청 지연을 없애려고 미리 한 번씩
    transcribe(np.zeros(16000, dtype=np.float32))
    tts.synthesize("안녕하세요.", voice_style=styles["F1"], lang="ko")

    app = FastAPI(title="csagent-voice")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "stt": "whisper-large-v3-turbo int8_float16 cuda",
                "tts": "supertonic-3 cpu", "tts_sample_rate": tts.sample_rate, "voices": VOICES}

    @app.post("/stt")
    async def post_stt(request: Request, degrade: str = "") -> dict:
        body = await request.body()
        if not body or len(body) % 2:
            raise HTTPException(422, "PCM16 바이트가 아닙니다")
        pcm = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
        audio_s = len(pcm) / 16000
        t = time.perf_counter()
        if degrade == "tel":
            import torch
            import torchaudio.functional as AF
            w8, _ = to_telephone(pcm, 16000)
            pcm = AF.resample(torch.from_numpy(w8.astype(np.float32)), 8000, 16000).numpy()
        with stt_lock:
            text = transcribe(pcm)
        return {"text": text, "ms": round((time.perf_counter() - t) * 1000), "audio_s": round(audio_s, 2)}

    @app.post("/tts")
    def post_tts(req: TTSRequest) -> Response:
        if not req.text.strip():
            raise HTTPException(422, "빈 문장")
        if req.voice not in styles:
            raise HTTPException(422, f"목소리는 {VOICES} 중 하나")
        t = time.perf_counter()
        with tts_lock:
            wav, _ = tts.synthesize(req.text, voice_style=styles[req.voice], lang="ko", total_steps=req.steps)
        gen_ms = round((time.perf_counter() - t) * 1000)
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        buf = io.BytesIO()
        sf.write(buf, wav, tts.sample_rate, format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav",
                        headers={"X-Gen-Ms": str(gen_ms), "X-Audio-S": f"{len(wav) / tts.sample_rate:.2f}"})

    return app


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8013)
    ap.add_argument("--threads", type=int, default=None, help="Supertonic ONNX intra-op 스레드")
    args = ap.parse_args()
    uvicorn.run(create_app(args.threads), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
