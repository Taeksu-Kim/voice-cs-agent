"""시연 영상 녹화 — 실제 웹 음성 통화를 Playwright로 진행하고 화면 + 소리를 mp4로 남긴다. 마이크 없이 고객 말은 TTS로 만든다.

    <playwright 있는 python> scripts/demo_record.py --out demos/card_lost.mp4 [--headed]

동작:
- 고객 말: 음성 서버(/tts)로 미리 합성 (상담원과 다른 목소리)
- 가짜 마이크: 페이지의 getUserMedia를 바꿔치기해, 합성한 고객 음성을 마이크 입력으로 넣는다 → VAD·STT·에이전트까지 실제 경로 그대로
- 턴 넘기기: 상담원 음성 재생이 끝나고 잠시 조용하면 다음 고객 말 (상담원이 한 말에 맞춰 대사를 고른다)
- 소리: 페이지 안에서 상담원 재생음 + 고객 음성을 섞어 녹음 → ffmpeg로 Playwright 화면 녹화와 합친다
웹 서버(scripts/run_server.py, 8080)·음성 서버(8013)·모델 서버가 떠 있어야 한다. 이 스크립트는 에이전트 코드와 무관하다 (playwright만 있으면 된다).
"""

from __future__ import annotations

import argparse
import base64
import re
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = "http://localhost:8080"
TTS = "http://127.0.0.1:8013/tts"

# 상담원의 마지막 문장(보통 질문)에 맞춰 고르는 고객 대사 — 위에서부터 맞는 것. 시나리오: 김민지 · 토스유스카드 분실
# 짧은 「네」는 음성 인식이 턴으로 안 잡을 수 있어 한 마디 더 붙인다
FIRST = "토스유스카드를 잃어버린 것 같아요. 누가 쓸까 봐 걱정돼요. 막아 주세요."
DONE = "아니요, 없어요. 감사합니다."
REPLIES = [
    (("다른 도움", "더 도와", "다른 문의", "더 궁금", "필요하시면"), DONE),
    (("찾을 수", "일시정지", "분실신고 중", "어느 쪽"), "어디서 잃어버렸는지 모르겠어요. 분실신고 해 주세요."),
    (("진행할까요", "할까요", "하시겠어요", "드릴까요"), "네, 분실신고로 진행해 주세요."),
    (("인증 요청", "인증을 보", "앱에서 인증", "앱으로 인증"), "네, 앱에서 인증했어요."),
    (("잃어버리신", "맞으세요", "맞나요", "맞으신가요", "이 카드"), "네, 그 카드 맞아요."),
]
FALLBACK = "네, 그렇게 해 주세요."
FAREWELL = ("좋은 하루", "통화를 종료", "감사합니다.")
END_MARKERS = ("통화 종료", "연결 종료")

INIT_JS = r"""
(() => {
  const Orig = window.AudioContext;
  window.__agentActive = 0; window.__agentCount = 0; window.__lastAgentEnd = 0;
  let mic = null;
  // 녹음용 섞개: 모든 재생 소리 + 고객 소리
  const mix = new Orig({ sampleRate: 48000 });
  const mixDest = mix.createMediaStreamDestination();
  window.__rec = { chunks: [], start: 0 };
  const rec = new MediaRecorder(mixDest.stream, { mimeType: "audio/webm;codecs=opus" });
  rec.ondataavailable = (e) => e.data.size && window.__rec.chunks.push(e.data);
  rec.start(1000); window.__rec.start = Date.now(); window.__recorder = rec;
  const tapped = new WeakSet();
  const tap = (ctx) => {
    if (tapped.has(ctx)) return ctx.__tap;
    const d = ctx.createMediaStreamDestination();
    mix.createMediaStreamSource(d.stream).connect(mixDest);
    tapped.add(ctx); ctx.__tap = d; return d;
  };
  const conn = AudioNode.prototype.connect;
  AudioNode.prototype.connect = function (t, ...r) {
    const out = conn.call(this, t, ...r);
    if (t instanceof AudioDestinationNode) conn.call(this, tap(t.context));
    return out;
  };
  const start = AudioBufferSourceNode.prototype.start;
  AudioBufferSourceNode.prototype.start = function (...a) {
    if (!mic || this.context !== mic.ctx) {
      window.__agentActive++; window.__agentCount++;
      this.addEventListener("ended", () => { window.__agentActive--; window.__lastAgentEnd = performance.now(); });
    }
    return start.apply(this, a);
  };
  // 서버 메시지 엿듣기: latency = 상담원 턴 끝 (session.py), hangup = 서버가 끊음
  window.__turnsDone = 0; window.__hangup = false;
  const WS = window.WebSocket;
  window.WebSocket = function (...args) {
    const ws = new WS(...args);
    ws.addEventListener("message", (e) => {
      if (typeof e.data !== "string") return;
      try { const m = JSON.parse(e.data); if (m.type === "latency") window.__turnsDone++; if (m.type === "hangup") window.__hangup = true; } catch {}
    });
    ws.addEventListener("close", () => { window.__hangup = true; });
    return ws;
  };
  window.WebSocket.prototype = WS.prototype;
  Object.assign(window.WebSocket, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
  navigator.mediaDevices.getUserMedia = async () => {
    if (!mic) { const ctx = new Orig({ sampleRate: 48000 }); mic = { ctx, dest: ctx.createMediaStreamDestination() }; }
    return mic.dest.stream;
  };
  window.__say = async (b64) => {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer;
    const buf = await mic.ctx.decodeAudioData(bytes);
    const src = mic.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(mic.dest);
    src.connect(tap(mic.ctx));
    src.start();
    return buf.duration;
  };
  window.__stopRec = () => new Promise((ok) => {
    window.__recorder.onstop = async () => {
      const blob = new Blob(window.__rec.chunks, { type: "audio/webm" });
      const b = new Uint8Array(await blob.arrayBuffer());
      let s = ""; for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode(...b.subarray(i, i + 0x8000));
      ok({ b64: btoa(s), start: window.__rec.start });
    };
    window.__recorder.stop();
  });
})();
"""


def tts(text: str, voice: str) -> str:
    req = urllib.request.Request(TTS, data=json.dumps({"text": text, "voice": voice, "steps": 8}).encode(),
                                 headers={"Content-Type": "application/json"})
    return base64.b64encode(urllib.request.urlopen(req, timeout=60).read()).decode()


def last_sentence(text: str) -> str:
    parts = [x for x in re.split(r"(?<=[.?!])\s+", text.strip()) if x]
    return parts[-1] if parts else text


def pick(agent_text: str, used: set[str]) -> str:
    """상담원의 마지막 문장을 먼저 보고, 없으면 전체를 본다."""
    for scope in (last_sentence(agent_text), agent_text):
        for keys, line in REPLIES:
            if any(k in scope for k in keys):
                return line
    return FALLBACK


def wait_greeting(page, timeout_s: float = 30) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = page.evaluate("({a: window.__agentActive, c: window.__agentCount, q: performance.now() - window.__lastAgentEnd})")
        if st["c"] > 0 and st["a"] == 0 and st["q"] > 1000:
            return
        time.sleep(0.2)


def wait_agent_done(page, turns_before: int, timeout_s: float = 60) -> bool:
    """서버가 턴을 마쳤고(latency) 그 턴 소리를 다 재생했으면 True. 통화가 끝났으면 False."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = page.evaluate("({t: window.__turnsDone, a: window.__agentActive, q: performance.now() - window.__lastAgentEnd,"
                           " h: window.__hangup})")
        if st["t"] > turns_before and st["a"] == 0 and st["q"] > 800:
            return not st["h"]
        if st["h"] and st["a"] == 0:
            return False
        time.sleep(0.2)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demos/card_lost.mp4")
    ap.add_argument("--voice", default="M2", help="고객 목소리 (상담원은 F1)")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--shots", help="통화 끝 화면·모니터 화면 캡처 폴더 (main.png · monitor.png · combined.png)")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [FIRST, FALLBACK] + [l for _, l in REPLIES]
    clips = {l: tts(l, a.voice) for l in dict.fromkeys(lines)}
    print("고객 음성", len(clips), "개 합성", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="demo_"))
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not a.headed, args=["--autoplay-policy=no-user-gesture-required"])
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, record_video_dir=str(tmp),
                                  record_video_size={"width": 1440, "height": 900}, permissions=["microphone"])
        ctx.add_init_script(INIT_JS)
        t_video = time.time()
        page = ctx.new_page()
        page.goto(WEB)
        page.wait_for_selector(".caller")
        page.click('.caller[data-caller="live_kim_minji"]')
        page.click('.sit[data-sit="card_lost"]')
        page.select_option("#input-mode", "voice")
        time.sleep(1.0)
        page.click("#start")
        wait_greeting(page)   # 고정 인사
        used: set[str] = set()
        line = FIRST
        for turn in range(a.max_turns):
            before = page.evaluate("window.__turnsDone")
            print(f"[고객] {line}", flush=True)
            page.evaluate("(b) => window.__say(b)", clips[line])
            used.add(line)
            if not wait_agent_done(page, before):
                break
            agent = page.locator(".msg.agent").last.inner_text().split("\n")[0]
            print(f"[상담원] {agent}", flush=True)
            if line == DONE or any(k in last_sentence(agent) for k in FAREWELL):
                break
            line = pick(agent, used)
        time.sleep(3.0)
        if a.shots:
            shots = Path(a.shots)
            shots.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(shots / "main.png"))
            mon = ctx.new_page()
            mon.goto(f"{WEB}/monitor.html?source=logs/{time.strftime('%Y%m%d')}")
            mon.wait_for_selector("#tiles .tile, #tiles > *", timeout=15000)
            time.sleep(2.0)
            mon.screenshot(path=str(shots / "monitor.png"))
            mon.close()
        res = page.evaluate("window.__stopRec()")
        video = Path(page.video.path())
        ctx.close()
        browser.close()

    audio = tmp / "audio.webm"
    audio.write_bytes(base64.b64decode(res["b64"]))
    offset = max(0.0, res["start"] / 1000 - t_video)   # 녹음 시작이 화면 녹화 시작보다 늦은 만큼
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-itsoffset", f"{offset:.3f}", "-i", str(audio),
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    str(out)], check=True)
    print(f"영상: {out}  (소리 오프셋 {offset:.2f}s)")
    if a.shots:
        shots = Path(a.shots)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(shots / "main.png"), "-i", str(shots / "monitor.png"),
                        "-filter_complex", "[0][1]vstack=inputs=2", str(shots / "combined.png")], check=True)
        print(f"캡처: {shots}/main.png · monitor.png · combined.png")


if __name__ == "__main__":
    main()
