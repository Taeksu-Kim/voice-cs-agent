"""브라우저 없이 음성 통화 전 구간 검증 — 고객 대사를 TTS(M1)로 합성해 실시간 속도로 WebSocket에 흘린다.

    ~/miniconda3/envs/voice-eval/bin/python scripts/voice_ws_test.py --scenario s1_b0_tracking \
        --line "택배가 아직 안 왔어요" --line "네 그거요" [--barge-in 1500] [--degrade tel]

- 웹 서버(scripts/run_server.py)와 음성 서버(serve.sh)가 떠 있어야 한다. 주소는 .env의 VOICE_URL과 --base
- 각 대사: 합성 → 16kHz → 20ms 프레임을 실시간 간격으로 전송 → 침묵 1초 → 그 턴의 latency 메시지까지 기다림
- 에이전트 음성은 재생하지 않고 길이만큼 「재생 중」(playing on/off)을 서버에 알린다 (브라우저 흉내)
- --barge-in MS: 첫 턴의 첫 음성을 받고 MS 뒤에 다음 대사를 보내 끼어들기를 시험한다
결과는 runs/voice/ws_<시각>.json 에도 남긴다.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import websocket
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
FRAME = 320   # 20ms @16k


def env(key: str) -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    return os.environ.get(key, "")


def synth16k(voice_url: str, text: str) -> np.ndarray:
    r = requests.post(f"{voice_url}/tts", json={"text": text, "voice": "M1"}, timeout=60)
    r.raise_for_status()
    w, sr = sf.read(io.BytesIO(r.content))
    return np.clip(resample_poly(w, 160, sr // 100), -1, 1).astype(np.float32)


class Call:
    def __init__(self, ws_url: str) -> None:
        self.ws = websocket.create_connection(ws_url, timeout=120)
        self.msgs: list[dict] = []
        self.cv = threading.Condition()
        self.t0 = time.time()
        self.play_until = self.t0
        threading.Thread(target=self._recv, daemon=True).start()

    def _recv(self) -> None:
        pending_audio = None
        while True:
            try:
                op, data = self.ws.recv_data()
            except Exception:
                return
            now = time.time()
            if op == websocket.ABNF.OPCODE_CLOSE:
                return
            if op == websocket.ABNF.OPCODE_BINARY:
                dur = sf.info(io.BytesIO(data)).duration
                start = max(now, self.play_until)
                self.play_until = start + dur
                self.ws.send(json.dumps({"type": "playing", "on": True}))
                threading.Timer(self.play_until - now, self._maybe_stop_playing).start()
                pending_audio["audio_s"] = round(dur, 2)
                continue
            try:
                msg = json.loads(data)
            except ValueError:
                print(f"!! 해석 못 한 프레임 op={op} len={len(data)} head={data[:24]!r}")
                continue
            msg["_t"] = round(now - self.t0, 3)
            if msg["type"] == "audio":
                pending_audio = msg
            if msg["type"] == "stop":
                self.play_until = now
                self.ws.send(json.dumps({"type": "playing", "on": False}))
            with self.cv:
                self.msgs.append(msg)
                self.cv.notify_all()

    def _maybe_stop_playing(self) -> None:
        if time.time() >= self.play_until - 0.05:
            print(f"  (재생 끝 {time.time() - self.t0:.2f}s)")
            try:
                self.ws.send(json.dumps({"type": "playing", "on": False}))
            except Exception:
                pass

    def stream(self, pcm: np.ndarray, silence_s: float = 1.0) -> None:
        pcm = np.concatenate([pcm, np.zeros(int(silence_s * 16000), dtype=np.float32)])
        t = time.time()
        for i in range(0, len(pcm), FRAME):
            self.ws.send_binary((pcm[i:i + FRAME] * 32767).astype("<i2").tobytes())
            t += FRAME / 16000
            time.sleep(max(0.0, t - time.time()))

    def wait(self, pred, start: int = 0, timeout: float = 120) -> dict:
        end = time.time() + timeout
        with self.cv:
            while True:
                for m in self.msgs[start:]:
                    if pred(m):
                        return m
                left = end - time.time()
                if left <= 0:
                    raise TimeoutError([m["type"] for m in self.msgs[start:]])
                self.cv.wait(left)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=f"http://{env('LLM_URL').split('//')[1].split(':')[0]}:8080")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--line", action="append", required=True, help="고객 대사 (여러 번)")
    ap.add_argument("--barge-in", type=int, default=None, help="첫 음성 수신 후 이 ms 뒤에 다음 대사로 끼어들기")
    ap.add_argument("--degrade", default="", help="tel: 서버가 전사 전에 전화 대역으로 열화")
    args = ap.parse_args()
    voice_url = env("VOICE_URL")

    sess = requests.post(f"{args.base}/api/sessions", json={"scenario_id": args.scenario, "mode": "voice"},
                         timeout=10).json()
    sid = sess["session_id"]
    ws_url = args.base.replace("http", "ws", 1) + f"/api/sessions/{sid}/voice?degrade={args.degrade}"
    pcms = [synth16k(voice_url, line) for line in args.line]   # 합성은 통화 전에 미리 (통화 지연에 안 섞이게)
    call = Call(ws_url)

    for i, (line, pcm) in enumerate(zip(args.line, pcms)):
        mark = len(call.msgs)
        print(f"\n▶ 고객 {i + 1}: {line} ({len(pcm) / 16000:.1f}s) — 전송 시작 {time.time() - call.t0:.2f}s"
              f" / 재생 끝 예정 {call.play_until - call.t0:.2f}s")
        call.stream(pcm)
        if args.barge_in is not None and i == 0 and i + 1 < len(pcms):
            call.wait(lambda m: m["type"] == "audio", mark)
            time.sleep(args.barge_in / 1000)
            continue          # 다음 대사를 바로 흘린다 (재생 중 끼어들기)
        # 턴마다 latency가 하나씩 온다 (끼어들기로 닫힌 턴 포함) — 대사 수만큼 모일 때까지
        call.wait(lambda m: sum(x["type"] == "latency" for x in call.msgs) >= i + 1)
        time.sleep(max(0.0, call.play_until - time.time()) + 0.3)   # 「다 듣고」 다음 말

    time.sleep(0.5)
    call.ws.send(json.dumps({"type": "hangup"}))
    call.ws.close()

    for m in call.msgs:
        t = m["_t"]
        if m["type"] == "stt":
            print(f"[{t:7.2f}] STT  {m['text']!r} ({m['ms']}ms, {m['audio_s']}s)")
        elif m["type"] == "audio":
            print(f"[{t:7.2f}] TTS  #{m['seq']} {m['text']} (합성 {m['gen_ms']}ms, {m.get('audio_s')}s)")
        elif m["type"] == "agent" and m["event"] in ("tool_call", "done", "error"):
            d = m["data"]
            print(f"[{t:7.2f}] {m['event']:5s}", d.get("name") or d.get("text") or d.get("message"))
        elif m["type"] in ("stop", "latency", "error"):
            print(f"[{t:7.2f}] {m['type']:5s}", {k: v for k, v in m.items() if k not in ("type", "_t")})
    out = ROOT / "runs" / "voice" / f"ws_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"session_id": sid, "scenario": args.scenario, "lines": args.line,
                               "barge_in": args.barge_in, "degrade": args.degrade, "messages": call.msgs},
                              ensure_ascii=False, indent=1))
    print("\n→", out.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
